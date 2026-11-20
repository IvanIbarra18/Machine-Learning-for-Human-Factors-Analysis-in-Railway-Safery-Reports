import argparse
import json
import numpy as np
import torch
import random
import time
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from data.data_loading import AVAILABLE_DATASETS, load_splits
from data.preprocessing import build_preprocessor
from rtdl_revisiting_models import FTTransformer
import joblib
import pandas as pd
from utilis.loss import ClassificationLoss
from tqdm import tqdm
from eval import evaluate
from utilis.metrics import score_event
from utilis.results import RESULTS_DIR, new_results_dir, seeds_summary

def parse_args():
    parser = argparse.ArgumentParser(description="Train one FT-Transformer that predicts all six events, then score it on the test set")
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60, metavar='N',help='maximum number of epochs to train (default: 60)')
    parser.add_argument("--num-workers", "-j", type=int, default=0)
    parser.add_argument("--precision",type=lambda value: value.upper(),choices=["FP32", "FP16", "BF16"],default="FP32",help="Requested training precision; AMP will be used in the training loop.",)
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[42],help="One model is trained per seed. With several seeds their probabilities are averaged.",)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--loss', choices=['bce', 'focal'], default='bce')
    parser.add_argument("--experiment",type=str,default="tabular_only",help="Name of this experiment. Results go to results/<name>/, or to a new numbered folder if that one already has results.",)
    parser.add_argument("--results-dir",type=Path,default=RESULTS_DIR,help="Folder that holds the experiment folders (default: results/).",)
    parser.add_argument("--output",type=Path,default=None,help="Directory for saved model checkpoints (default: checkpoints/<experiment folder>).",)
    parser.add_argument("--dataset",type=str,choices=AVAILABLE_DATASETS,required=True,help="Processed CSV dataset to load.",)

    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must all be different")

    return args


def build_datasets(parts):
    """Fit the input preprocessing on training only, then turn every split into tensors.

    parts maps "training" / "validation" / "test" to (features, 0/1 label table).
    Returns the fitted preprocessor, one TensorDataset per split (continuous
    inputs, categorical inputs, labels) and the size of each categorical input.
    """
    X_train = parts["training"][0]

    # Fit only on training data to prevent data leakage.
    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    # Keep the FT notebook's separate continuous and categorical inputs.
    cont_slice = preprocessor.output_indices_["numeric"]
    cat_slice = preprocessor.output_indices_["categorical"]
    encoder = preprocessor.named_transformers_["categorical"]
    cat_cardinalities = [len(c) + 1 for c in encoder.categories_]

    datasets = {}
    for split, (X, Y) in parts.items():
        processed = np.asarray(preprocessor.transform(X), dtype=np.float32)

        if not np.isfinite(processed).all():
            raise ValueError(f"{split.capitalize()} features contain NaN or infinity.")

        datasets[split] = TensorDataset(
            torch.from_numpy(processed[:, cont_slice]),
            torch.from_numpy(processed[:, cat_slice].astype(np.int64) + 1),
            torch.from_numpy(Y.to_numpy(dtype=np.float32, copy=True)),
        )

    return preprocessor, datasets, cat_cardinalities


def train_one_seed(seed, args, device, datasets, n_cont_features, cat_cardinalities, label_names, feature_columns):
    """Train one model, keep the epoch with the best validation mean AP, and
    return that model's validation and test probabilities."""

    # Set random seeds.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure DataLoaders.
    pin_memory = device.type == "cuda"

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }

    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(datasets["training"], shuffle=True, **loader_options)
    val_loader = DataLoader(datasets["validation"], shuffle=False, **loader_options)
    test_loader = DataLoader(datasets["test"], shuffle=False, **loader_options)

    # Initialize the model: one output (logit) per event.
    model_kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=len(label_names),
        **model_kwargs,
    ).to(device)

    # Configure automatic mixed precision.
    use_amp = args.precision != "FP32" and device.type == "cuda"

    amp_dtype = {"FP32": torch.float32,"FP16": torch.float16,"BF16": torch.bfloat16,}[args.precision]

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp and amp_dtype == torch.float16,)

    # Initialize the loss function. Each event is its own yes/no question.
    criterion = ClassificationLoss()
    if args.loss == "focal":
        from torch_focalloss import BinaryFocalLoss
        criterion = BinaryFocalLoss(gamma=2.0, reduction="mean")

    # Initialize the optimizer.
    optimizer = model.make_default_optimizer()

    checkpoint_path = args.output / f"ft_transformer_seed{seed}.pt"
    best_ap, best_epoch = -1.0, 0
    best_state, best_val_probabilities = None, None
    history = []

    # Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples_seen = 0

        epoch_start = time.perf_counter()

        # disable=None hides the bar when output is not a terminal (for example a log file).
        with tqdm(total=len(train_loader.dataset),desc=f"Seed {seed} epoch {epoch}/{args.epochs}",unit="sample",dynamic_ncols=True,disable=None,) as pbar:
            for x_cont, x_cat, targets in train_loader:

                x_cont = x_cont.to(device, non_blocking=pin_memory)
                x_cat = x_cat.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)

                optimizer.zero_grad(set_to_none=True)

                # Forward pass.
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp,):
                    logits = model(x_cont, x_cat)

                # Calculate loss in FP32 for numerical stability.
                loss = criterion(logits.float(), targets)

                # Backward pass and optimizer update.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = targets.size(0)

                train_loss_sum += loss.item() * batch_size
                train_samples_seen += batch_size

                elapsed = time.perf_counter() - epoch_start
                samples_per_second = train_samples_seen / max(elapsed, 1e-9)

                pbar.update(batch_size)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    samples_s=f"{samples_per_second:.2f}",
                )

        avg_train_loss = train_loss_sum / train_samples_seen

        # Validation is handled by eval.py.
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            desc=f"Validation {epoch}/{args.epochs}",
        )

        val_mean_ap = val_metrics["mean_ap"]

        history.append(dict(epoch=epoch, train_loss=avg_train_loss,
                            val_loss=val_metrics["loss"], val_mean_ap=val_mean_ap,
                            **{f"val_ap: {name}": ap for name, ap in zip(label_names, val_metrics["ap"])}))
        pd.DataFrame(history).to_csv(args.output / f"history_seed{seed}.csv", index=False)

        # Keep the epoch with the best validation mean AP (no cutoff needed, same measure the baselines select on).
        if val_mean_ap > best_ap:
            best_ap, best_epoch = val_mean_ap, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_probabilities = val_metrics["probabilities"]
            torch.save({
                "model_state_dict": best_state,
                "model_kwargs": model_kwargs,
                "n_cont_features": n_cont_features,
                "cat_cardinalities": cat_cardinalities,
                "label_names": label_names,
                "feature_columns": feature_columns,
                "epoch": epoch, "val_mean_ap": best_ap,
                "loss": args.loss, "seed": seed,
            }, checkpoint_path)

        print(
            f"Seed {seed} epoch {epoch}/{args.epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Validation Loss: {val_metrics['loss']:.4f} | "
            f"Validation Mean AP: {val_mean_ap:.4f} | "
            f"{time.perf_counter() - epoch_start:.0f}s",
            flush=True,
        )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping; best epoch {best_epoch}, validation mean AP {best_ap:.4f}")
            break

    # The test set is used once per trained model, after its checkpoint was chosen on validation.
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        desc="Test",
    )
    print(f"Seed {seed} checkpoint: {checkpoint_path} (epoch {best_epoch}, validation mean AP {best_ap:.4f})\n", flush=True)

    return best_val_probabilities, test_metrics["probabilities"], best_epoch


def score_on_test(model_name, params, label_names, Y_val, P_val, Y_test, P_test):
    """One row per event, in the same columns as scripts/run_baselines.py writes."""
    return pd.DataFrame([
        dict(setting="Structured", model=model_name, label=label, params=params,
             **score_event(Y_val[:, j], P_val[:, j], Y_test[:, j], P_test[:, j]),
             n_test=len(Y_test))
        for j, label in enumerate(label_names)
    ])


def main():
    args = parse_args()

    # 1. Load the training, validation and test splits (each with one 0/1 column per event).
    parts = load_splits(dataset=args.dataset)
    X_train, Y_train = parts["training"]
    label_names = Y_train.columns.tolist()

    # 2. Build the preprocessing pipeline and the model inputs.
    preprocessor, datasets, cat_cardinalities = build_datasets(parts)

    n_cont_features = datasets["training"].tensors[0].shape[1]
    n_cat_features = datasets["training"].tensors[1].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Print run configuration.
    print(f"Dataset: {args.dataset}")
    print(f"Events: {', '.join(label_names)}")
    print(f"FT-Transformer: {n_cont_features} numeric, {n_cat_features} categorical features")
    print(f"Loss: {args.loss}; optimizer: model.make_default_optimizer()")
    print(f"Training / validation / test samples: {len(datasets['training'])} / {len(datasets['validation'])} / {len(datasets['test'])}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: up to {args.epochs}, patience {args.patience}")
    print(f"Seeds: {args.seeds}")
    print(f"Requested precision: {args.precision}")
    print(f"Device: {device}\n", flush=True)

    # 4. Results go to this experiment's own folder and are never overwritten:
    # if results/<experiment>/ already has results, a new numbered folder is used.
    # Checkpoints go to the folder of the same name under checkpoints/.
    results_dir = new_results_dir(args.experiment, args.results_dir)
    if args.output is None:
        args.output = Path("checkpoints") / results_dir.relative_to(args.results_dir)
    print(f"Results folder: {results_dir}")
    print(f"Checkpoint folder: {args.output}\n", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, args.output / "preprocessor.joblib")

    # 5. Train one model per seed. Each model is scored on the test set the same way
    # scripts/run_baselines.py does (the yes/no cutoff of each event is picked on
    # validation, then applied to test) and its scores are saved as soon as it is done.
    Y_val = parts["validation"][1].to_numpy()
    Y_test = parts["test"][1].to_numpy()

    val_probabilities, test_probabilities, best_epochs, seed_results = [], [], [], []
    for seed in args.seeds:
        p_val, p_test, best_epoch = train_one_seed(
            seed, args, device, datasets, n_cont_features, cat_cardinalities,
            label_names, X_train.columns.tolist(),
        )
        val_probabilities.append(p_val)
        test_probabilities.append(p_test)
        best_epochs.append(best_epoch)

        params = json.dumps(dict(seed=seed, best_epoch=best_epoch, loss=args.loss, batch_size=args.batch_size))
        seed_results.append(score_on_test("FTTransformer", params, label_names, Y_val, p_val, Y_test, p_test))
        seed_results[-1].to_csv(results_dir / f"seed_{seed}.csv", index=False)

    # 6. With several seeds, the final prediction is the average probability over the models.
    n_seeds = len(args.seeds)
    final, average = seed_results[0], None
    if n_seeds > 1:
        params = json.dumps(dict(seeds=args.seeds, best_epochs=best_epochs, loss=args.loss, batch_size=args.batch_size))
        average = score_on_test(f"FTTransformer ({n_seeds}-seed average)", params, label_names,
                                Y_val, np.mean(val_probabilities, axis=0), Y_test, np.mean(test_probabilities, axis=0))
        average.to_csv(results_dir / "average.csv", index=False)
        final = average

    with open(args.output / "thresholds.json", "w") as f:
        json.dump(dict(model=final.model.iloc[0], seeds=args.seeds, thresholds=dict(zip(label_names, final.threshold))), f, indent=2)

    # 7. The summary is a small markdown file, so it can be shared and kept in git.
    summary = seeds_summary(
        f"{results_dir.name}: FT-Transformer on the structured fields",
        [f"One model per seed. Seeds {args.seeds}; best epoch per seed {best_epochs}; "
         f"loss {args.loss}; batch size {args.batch_size}.",
         f"Checkpoints: {args.output.as_posix()} (not in git). Written {time.strftime('%Y-%m-%d %H:%M')}."],
        args.seeds, seed_results, average,
    )
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Results: {results_dir}")
    print(f"Summary to share: {results_dir / 'summary.md'}")
    print(f"Checkpoints: {args.output}")

if __name__ == "__main__":
    main()
