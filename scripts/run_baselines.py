# RQ1: To what extent can structured human-factor, environmental, personnel, and operational features classify the six
# recorded event categories in NASA C3RS reports?

# Trains 3 models (Logistic Regression, Random Forest, XGBoost) on the same data
# and saves their scores to results/tabular_only/baselines/ (seed_42.csv and summary.md).
# Results are never overwritten: a later run writes to baselines_2/, then baselines_3/, ...

# One model, quick test:
# python scripts/run_baselines.py --models LogReg
# python scripts/run_baselines.py --models RandomForest
# python scripts/run_baselines.py --models XGBoost

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from preprocessing_data import NOTEBOOK_DIR, build_features, make_labels, make_split, narrative_text
from utilis.metrics import score_event
from utilis.results import RESULTS_DIR, new_results_dir, summary_markdown

SEED = 42
MODELS = ("LogReg", "RandomForest", "XGBoost")

# The 70/15/15 split is make_split from preprocessing_data.py, the same one train.py's CSVs use.

# The settings we try for each model. The best one is picked on the validation set.
def param_grid(name, pos_weight):
    if name == "LogReg":
        return [dict(C=C, class_weight=w) for C in (0.1, 1.0) for w in (None, "balanced")]
    
    if name == "RandomForest":
        return [dict(min_samples_leaf=l, class_weight=w) for l in (1, 3) for w in (None, "balanced_subsample")]
    return [dict(max_depth=d, scale_pos_weight=w) for d in (4, 6) for w in (1.0, pos_weight)]


# Create a model with the given settings.
def make_model(name, p):
    if name == "LogReg":
        return LogisticRegression(max_iter=3000, solver="liblinear", **p)
    
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **p)
    return XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                         eval_metric="aucpr", n_jobs=-1, random_state=SEED, **p)


# Fill missing numbers with the median value from the training part,
# then scale every column to between -1 and 1.
def scale_inputs(X, tr):
    X = X.fillna(X.iloc[tr].median())
    return MaxAbsScaler().fit(X.iloc[tr]).transform(X)


# Train and score one model on all 6 event types (one yes/no model per event type).
def evaluate(name, Y, X, tr, va, te):
    rows = []
    for label in Y.columns:
        y = Y[label].values
        pos_weight = (y[tr] == 0).sum() / max(1, y[tr].sum())   # how many "no" per "yes"

        # 1. Try each setting
        # Keep the one with the best validation score.
        best = None
        for p in param_grid(name, pos_weight):
            m = make_model(name, p).fit(X[tr], y[tr])
            p_val = m.predict_proba(X[va])[:, 1]
            ap = average_precision_score(y[va], p_val)
            if best is None or ap > best[0]:
                best = (ap, m, p_val, p)
        _, m, p_val, p = best

        # 2. Pick the yes/no cutoff on validation, then
        # 3. score 1x on the test set (which the model has never seen).
        # score_event is shared with train.py, so the neural network is scored the same way.
        p_test = m.predict_proba(X[te])[:, 1]
        rows.append(dict(setting="Structured", model=name, label=label, params=json.dumps(p),
                         **score_event(y[va], p_val, y[te], p_test),
                         n_test=len(te)))
    return pd.DataFrame(rows)


def main():
    # Choose which models to run.
    ap = argparse.ArgumentParser(description="RQ1: compare 3 models on the coded fields")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--experiment", default="tabular_only/baselines",
                    help="Results go to results/<name>/, or to a new numbered folder if that one already has results.")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="Folder that holds the experiment folders (default: results/).")
    a = ap.parse_args()

    # Read the CSV and clean up column names.
    raw = (NOTEBOOK_DIR / os.getenv("DATA_PATH")).resolve()
    print(f"Reading {raw}")
    df = pd.read_csv(raw, low_memory=False)
    df.columns = [" ".join(str(c).split()) for c in df.columns]

    Y = make_labels(df)

    X = build_features(df)
    tr, va, te = make_split(Y, narrative_text(df))

    X = scale_inputs(X, tr)
    print(f"{len(df):,} reports, {X.shape[1]} features, " f"split {len(tr):,} / {len(va):,} / {len(te):,}")

    # Train and score each model.
    results = []
    for name in a.models:
        start = time.time()
        r = evaluate(name, Y, X, tr, va, te)
        results.append(r)
        print(f"{name:12s} average F1 {r.f1.mean():.3f}   mean AP {r.ap.mean():.3f}   " f"({time.time() - start:.0f}s)")

    # Results go to their own folder and are never overwritten: if the folder
    # already has results, a new numbered folder is used (see utilis/results.py).
    out = new_results_dir(a.experiment, a.results_dir)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(out / f"seed_{SEED}.csv", index=False)

    # The summary is a small markdown file, so it can be shared and kept in git.
    summary = summary_markdown(
        f"{out.name}: baselines on the structured fields",
        [f"{', '.join(a.models)}: one yes/no model per event, settings chosen per event by validation AP. Seed {SEED}.",
         f"Written {time.strftime('%Y-%m-%d %H:%M')}."],
        results,
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Saved {out}")
    print(f"Summary to share: {out / 'summary.md'}")

if __name__ == "__main__":
    main()