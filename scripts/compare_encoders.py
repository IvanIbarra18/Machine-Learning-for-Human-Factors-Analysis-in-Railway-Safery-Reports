# Which frozen text encoder gives the most useful report vectors?
#
# For each encoder:
#   1. join each report's narratives into one text and cut it to the encoder's length limit
#   2. run the frozen encoder once and average the token vectors into one vector per report
#   3. fit logistic regression per event on the training reports
#   4. report validation AP per event
#
# Only training and validation reports are used. The test reports are never tokenized,
# embedded or scored here.
#
# All models (about 15 minutes on the GPU the first time, including about 1 GB of downloads):
#   python scripts/compare_encoders.py
# Quick check on a few reports with one model:
#   python scripts/compare_encoders.py --models distilbert-base-uncased --limit 200

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing_data import NOTEBOOK_DIR, ROOT, make_labels, make_split, narrative_text
from utilis.metrics import average_precision_per_event
from utilis.results import RESULTS_DIR, markdown_table, new_results_dir

MODELS = ("bert-base-uncased", "distilbert-base-uncased", "roberta-base")
LONG_MODEL = "answerdotai/ModernBERT-base"    # reads longer texts; added only if many reports are long
SHORT_LIMIT = 512                             # the most tokens the three base models can read
REFERENCE_TOKENIZER = "bert-base-uncased"     # its token counts decide whether LONG_MODEL is added
CACHE_DIR = ROOT / "datasets" / "embeddings"  # report vectors are kept here (not in git) and reused


def token_counts(tokenizer, texts):
    """Number of tokens of each text, special tokens included, nothing cut off."""
    return np.array(tokenizer(texts, truncation=False, return_attention_mask=False, verbose=False,
                              return_length=True)["length"])


@torch.no_grad()
def embed(model_name, texts, max_length, batch_tokens, device):
    """One vector per text: the average of the frozen encoder's token vectors."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ModernBERT can try to compile itself, which needs tools Windows does not have; switch that off.
    extra = {"reference_compile": False} if "modernbert" in model_name.lower() else {}
    model = AutoModel.from_pretrained(model_name, **extra).to(device).eval()

    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    ids, masks = encoded["input_ids"], encoded["attention_mask"]

    # Longest texts first: texts of similar length share a batch, so little padding is
    # computed, and if the GPU runs out of memory it happens on the first batch.
    order = np.argsort([-len(i) for i in ids])
    vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)

    start = 0
    with tqdm(total=len(texts), desc=model_name, unit="report", dynamic_ncols=True, disable=None) as bar:
        while start < len(order):
            size = max(1, batch_tokens // len(ids[order[start]]))    # fewer texts per batch when they are long
            rows = order[start:start + size]
            batch = tokenizer.pad({"input_ids": [ids[r] for r in rows], "attention_mask": [masks[r] for r in rows]},
                                  return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state                # (texts, tokens, vector size)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vectors[rows] = ((hidden * mask).sum(dim=1) / mask.sum(dim=1)).float().cpu().numpy()    # padding is left out of the average
            start += size
            bar.update(len(rows))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return vectors


def cached_embed(model_name, split, texts, max_length, batch_tokens, device, use_cache):
    """Compute the report vectors of one split, or load them if this was done before."""
    path = CACHE_DIR / model_name.split("/")[-1] / f"{split}_{max_length}.npy"
    if use_cache and path.is_file():
        vectors = np.load(path)
        if len(vectors) == len(texts):
            print(f"  {split}: loaded {len(vectors):,} saved vectors from {path}")
            return vectors
    vectors = embed(model_name, texts, max_length, batch_tokens, device)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
    return vectors


def validation_ap(E_train, Y_train, E_val, Y_val):
    """Logistic regression per event on the training vectors; AP of each event on validation."""
    scaler = StandardScaler().fit(E_train)
    X_train, X_val = scaler.transform(E_train), scaler.transform(E_val)
    P_val = np.zeros(Y_val.shape)
    for j in range(Y_train.shape[1]):
        model = LogisticRegression(C=1.0, max_iter=3000).fit(X_train, Y_train[:, j])
        P_val[:, j] = model.predict_proba(X_val)[:, 1]
    return average_precision_per_event(Y_val, P_val)


def length_row(counts, limit):
    """How long the texts are for one tokenizer, and how many exceed its limit."""
    return {"Reports": len(counts), "Median tokens": int(np.median(counts)),
            "90% are under": int(np.percentile(counts, 90)), "99% are under": int(np.percentile(counts, 99)),
            "Longest": int(counts.max()), "Length limit": limit,
            "Over the limit": int((counts > limit).sum()), "Share over the limit": f"{(counts > limit).mean():.1%}"}


def main():
    ap = argparse.ArgumentParser(description="Compare frozen text encoders by validation AP")
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="Hugging Face model names.")
    ap.add_argument("--long-model", default=LONG_MODEL,
                    help="Added when more than --long-share of the reports exceed 512 tokens. Use 'none' to never add it.")
    ap.add_argument("--long-share", type=float, default=0.15)
    ap.add_argument("--long-limit", type=int, default=2048, help="Length limit, in tokens, used for the long model.")
    ap.add_argument("--batch-tokens", type=int, default=32 * SHORT_LIMIT,
                    help="Most tokens in one GPU batch, padding included. Lower it if the GPU runs out of memory.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Quick check: use only this many training reports (and half as many validation reports).")
    ap.add_argument("--experiment", default="encoder_comparison",
                    help="Results go to results/<name>/, or to a new numbered folder if that one already has results.")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    a = ap.parse_args()

    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()    # keep only this script's own progress bars
    warnings.filterwarnings("ignore", message="Token indices sequence length")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The shared labels and split. Narratives are the model input here, not only the grouping key.
    raw = (NOTEBOOK_DIR / os.getenv("DATA_PATH")).resolve()
    print(f"Reading {raw}")
    df = pd.read_csv(raw, low_memory=False)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    Y = make_labels(df)
    narratives = narrative_text(df)
    tr, va, _ = make_split(Y, narratives)    # the third part is the test rows: not used anywhere below
    if a.limit:
        tr, va = tr[:a.limit], va[:max(1, a.limit // 2)]
    texts = {"training": narratives.iloc[tr].tolist(), "validation": narratives.iloc[va].tolist()}
    Y_train, Y_val = Y.iloc[tr].to_numpy(), Y.iloc[va].to_numpy()
    events = Y.columns.tolist()
    all_texts = texts["training"] + texts["validation"]
    print(f"{len(tr):,} training and {len(va):,} validation reports; device {device}")

    # Results go to their own folder and are never overwritten (see utilis/results.py).
    run_dir = new_results_dir(a.experiment, a.results_dir)

    # 1. How many reports are longer than the base models can read?
    reference = token_counts(AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER), all_texts)
    share_long = float((reference > SHORT_LIMIT).mean())
    models = list(a.models)
    add_long = a.long_model.lower() != "none" and share_long > a.long_share and a.long_model not in models
    if add_long:
        models.append(a.long_model)
    count_note = (f"{int((reference > SHORT_LIMIT).sum()):,} of {len(reference):,} training and validation reports "
                  f"({share_long:.1%}) are longer than {SHORT_LIMIT} tokens ({REFERENCE_TOKENIZER} tokenizer). "
                  f"{a.long_model} is added above {a.long_share:.0%}: {'added' if add_long else 'not added'}.")
    print(count_note)

    lengths, scores, seconds = {}, {}, {}

    def write_summary():
        """Written after every model, so a run that stops early still leaves its results."""
        parts = [f"# {run_dir.name}: frozen text encoders, logistic regression on averaged token vectors", "",
                 "- Input: each report's narratives joined into one text, cut to the model's length limit.",
                 "- One vector per report (average of the token vectors of the frozen encoder), standardized, "
                 "then logistic regression (C=1.0) per event on the training reports. No random seed is involved.",
                 f"- Validation scores on {len(va):,} reports, trained on {len(tr):,}. The test set was not read.",
                 f"- {count_note}",
                 f"- Written {time.strftime('%Y-%m-%d %H:%M')}.", ""]
        if scores:
            table = pd.DataFrame(scores, index=events).rename_axis("Event")
            table.loc["Mean"] = table.mean()
            parts += ["## Validation AP per event", "", markdown_table(table), ""]
        parts += ["## Report length in tokens", "", markdown_table(pd.DataFrame(lengths).T.rename_axis("Tokenizer")), ""]
        if seconds:
            parts += ["## Time to compute the report vectors", "",
                      markdown_table(pd.DataFrame({"Seconds": seconds}).rename_axis("Model"), digits=0), ""]
        (run_dir / "summary.md").write_text("\n".join(parts), encoding="utf-8")

    lengths[REFERENCE_TOKENIZER] = length_row(reference, SHORT_LIMIT)
    write_summary()

    # 2. One encoder at a time: report vectors, logistic regression, validation AP.
    for name in models:
        limit = a.long_limit if name == a.long_model else SHORT_LIMIT
        short = name.split("/")[-1]
        print(f"\n{name} (length limit {limit} tokens)")
        if name != REFERENCE_TOKENIZER:
            lengths[short] = length_row(token_counts(AutoTokenizer.from_pretrained(name), all_texts), limit)

        start = time.time()
        vectors = {split: cached_embed(name, split, texts[split], limit, a.batch_tokens, device, use_cache=not a.limit)
                   for split in ("training", "validation")}
        seconds[short] = time.time() - start

        scores[short] = validation_ap(vectors["training"], Y_train, vectors["validation"], Y_val)
        print(f"  validation AP per event {np.round(scores[short], 3).tolist()}   mean {np.mean(scores[short]):.3f}")

        # Each model's results in its own folder.
        model_dir = run_dir / short
        model_dir.mkdir()
        pd.DataFrame({"model": name, "label": events, "validation_ap": scores[short], "length_limit": limit,
                      "n_train": len(tr), "n_validation": len(va)}).to_csv(model_dir / "validation.csv", index=False)
        own = pd.DataFrame({"Validation AP": scores[short]}, index=events).rename_axis("Event")
        own.loc["Mean"] = own.mean()
        (model_dir / "summary.md").write_text(
            f"# {short}: frozen encoder, validation AP\n\n"
            f"- {name}, length limit {limit} tokens; {lengths[short]['Share over the limit']} of reports are longer and were cut.\n"
            f"- Validation scores on {len(va):,} reports, trained on {len(tr):,}. The test set was not read.\n\n"
            f"{markdown_table(own)}\n", encoding="utf-8")
        write_summary()

    print(f"\n{(run_dir / 'summary.md').read_text(encoding='utf-8')}")
    print(f"Results: {run_dir}")
    print(f"Summary to share: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
