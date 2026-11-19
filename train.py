import argparse
import numpy as np
import torch
import random
import time
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from data.data_loading import AVAILABLE_DATASETS, encode_labels, load_train_validation
from data.preprocessing import build_preprocessor
from rtdl_revisiting_models import FTTransformer
import joblib
import pandas as pd
from utilis.loss import ClassificationLoss
from tqdm import tqdm
from eval import evaluate
from utilis.metrics import macro_f1

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare settings for training")
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 30)')
    parser.add_argument("--num-workers", "-j", type=int, default=0)
    parser.add_argument("--precision",type=lambda value: value.upper(),choices=["FP32", "FP16", "BF16"],default="FP32",help="Requested training precision; AMP will be used in the training loop.",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    parser.add_argument("--output",type=Path,default=Path("checkpoints"),help="Directory for saved model checkpoints.",)
    parser.add_argument("--dataset",type=str,choices=AVAILABLE_DATASETS,required=True,help="Processed CSV dataset to load.",)

    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    return args

def main():
    args = parse_args()

    # Set random seeds.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load training and validation splits.
    X_train, y_train, X_validation, y_validation = load_train_validation(
        dataset=args.dataset,
    )

    # 2. Build and apply the preprocessing pipeline.
    preprocessor = build_preprocessor(X_train)

    # Fit only on training data to prevent data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_validation)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_val_processed = np.asarray(X_val_processed, dtype=np.float32)

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Training features contain NaN or infinity.")

    if not np.isfinite(X_val_processed).all():
        raise ValueError("Validation features contain NaN or infinity.")

    # Keep the FT notebook's separate continuous and categorical inputs.
    cont_slice = preprocessor.output_indices_["numeric"]
    cat_slice = preprocessor.output_indices_["categorical"]
    x_cont_train = X_train_processed[:, cont_slice]
    x_cont_val = X_val_processed[:, cont_slice]
    x_cat_train = X_train_processed[:, cat_slice].astype(np.int64) + 1
    x_cat_val = X_val_processed[:, cat_slice].astype(np.int64) + 1
    encoder = preprocessor.named_transformers_["categorical"]
    cat_cardinalities = [len(c) + 1 for c in encoder.categories_]

    # Classes are learned from training only; unexpected validation labels fail.
    class_names = sorted(set(y_train))
    y_train_encoded = encode_labels(y_train, class_names)
    y_val_encoded = encode_labels(y_validation, class_names)

    y_train_encoded = y_train_encoded.astype(np.int64, copy=False)
    y_val_encoded = y_val_encoded.astype(np.int64, copy=False)

    train_dataset = TensorDataset(
        torch.from_numpy(x_cont_train),
        torch.from_numpy(x_cat_train),
        torch.from_numpy(y_train_encoded.copy()),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(x_cont_val),
        torch.from_numpy(x_cat_val),
        torch.from_numpy(y_val_encoded.copy()),
    )

    # 4. Configure DataLoaders.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = device.type == "cuda"

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }

    if args.num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )

    val_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    # 5. Initialize the model.
    # Number of target classes in the CSV.
    num_classes = len(class_names)

    model_kwargs = FTTransformer.get_default_kwargs()
    model = FTTransformer(
        n_cont_features=x_cont_train.shape[1],
        cat_cardinalities=cat_cardinalities,
        d_out=num_classes,
        **model_kwargs,
    ).to(device)

    # Configure automatic mixed precision.
    use_amp = args.precision != "FP32" and device.type == "cuda"

    amp_dtype = {"FP32": torch.float32,"FP16": torch.float16,"BF16": torch.bfloat16,}[args.precision]

    scaler = torch.amp.GradScaler("cuda",enabled=use_amp and amp_dtype == torch.float16,)

    # 6. Initialize the loss function.
    criterion = ClassificationLoss()
    if args.loss == "focal":
        from torch_focalloss import MultiClassFocalLoss
        criterion = MultiClassFocalLoss(gamma=2.0, reduction="mean")

    # 7. Initialize the optimizer.
    optimizer = model.make_default_optimizer()


    # 8. Print run configuration.
    print(f"Dataset: {args.dataset}")
    print(f"FT-Transformer: {x_cont_train.shape[1]} numeric, {x_cat_train.shape[1]} categorical features")
    print(f"Loss: {args.loss}; optimizer: model.make_default_optimizer()")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Requested precision: {args.precision}")
    print(f"Device: {device}")
    print(f"DataLoader workers: {args.num_workers}; pin_memory={pin_memory}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    args.output.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch = -1.0, 0
    history = []
    joblib.dump(preprocessor, args.output / "preprocessor.joblib")

    # 9. Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_samples_seen = 0
        train_targets = []
        train_predictions = []

        epoch_start = time.perf_counter()

        with tqdm(total=len(train_dataset),desc=f"Epoch {epoch}/{args.epochs}",unit="sample",dynamic_ncols=True,) as pbar:
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

                predictions = logits.argmax(dim=1)
                train_targets.extend(targets.detach().cpu().tolist())
                train_predictions.extend(predictions.detach().cpu().tolist())

                elapsed = time.perf_counter() - epoch_start
                samples_per_second = train_samples_seen / max(elapsed, 1e-9)

                pbar.update(batch_size)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    samples_s=f"{samples_per_second:.2f}",
                )

        avg_train_loss = train_loss_sum / train_samples_seen

        train_macro_f1 = macro_f1(
            y_true=train_targets,
            y_pred=train_predictions,
            num_classes=num_classes,
        )

        # 10. Validation is handled by eval.py.
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            desc=f"Validation {epoch}/{args.epochs}",
        )

        avg_val_loss = val_metrics["loss"]
        val_macro_f1 = val_metrics["macro_f1"]

        history.append(dict(epoch=epoch, train_loss=avg_train_loss,
                            train_macro_f1=train_macro_f1, val_loss=avg_val_loss,
                            val_macro_f1=val_macro_f1))
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        if val_macro_f1 > best_f1:
            best_f1, best_epoch = val_macro_f1, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "n_cont_features": x_cont_train.shape[1],
                "cat_cardinalities": cat_cardinalities,
                "class_names": class_names,
                "feature_columns": X_train.columns.tolist(),
                "epoch": epoch, "val_macro_f1": best_f1,
                "loss": args.loss,
            }, args.output / "best_ft_transformer.pt")

        print(
            f"Epoch {epoch}/{args.epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Macro F1: {train_macro_f1:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"Validation Macro F1: {val_macro_f1:.4f}\n"
        )

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping; best epoch {best_epoch}, Macro F1 {best_f1:.4f}")
            break
    print(f"Best checkpoint: {args.output / 'best_ft_transformer.pt'}")

if __name__ == "__main__":
    main()
