from pathlib import Path
from typing import Literal

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"
AVAILABLE_DATASETS = ("A",)
# Each event has its own 0/1 column in the processed CSVs, named "label: <event>".
LABEL_PREFIX = "label: "
SPLITS = ("training", "validation", "test")


def get_processed_dataset_path(
    dataset: str,
    split: Literal["training", "validation", "test"],
) -> Path:
    if dataset not in AVAILABLE_DATASETS:
        valid = ", ".join(AVAILABLE_DATASETS)
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of: {valid}.")
    if split not in SPLITS:
        valid = ", ".join(SPLITS)
        raise ValueError(f"Unknown split {split!r}. Choose one of: {valid}.")

    csv_path = PROCESSED_DIR / dataset / f"{dataset}_{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Processed dataset CSV not found: {csv_path}.")

    return csv_path


def load_processed_split(
    dataset: str,
    split: Literal["training", "validation", "test"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the features and the 0/1 label table (one column per event) of one split."""
    csv_path = get_processed_dataset_path(dataset, split)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    label_columns = [c for c in df.columns if c.startswith(LABEL_PREFIX)]
    if not label_columns:
        raise ValueError(
            f"No {LABEL_PREFIX!r} columns in {csv_path}. "
            "Regenerate the processed CSVs with preprocessing_data.py."
        )
    if df.empty:
        raise ValueError(f"Processed dataset CSV is empty: {csv_path}.")

    Y = df[label_columns]
    if not Y.isin([0, 1]).all().all():
        raise ValueError(f"Labels in {csv_path} must be 0 or 1 with none missing.")
    Y = Y.astype("int64")
    Y.columns = [c[len(LABEL_PREFIX):] for c in label_columns]

    X = df.drop(columns=label_columns)
    X.columns = X.columns.str.lower()
    if X.columns.duplicated().any():
        duplicates = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names in {csv_path}: {duplicates}.")
    if X.shape[1] == 0:
        raise ValueError(f"No feature columns found in {csv_path}.")

    return X, Y


def load_splits(dataset: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load training, validation and test, checking that their columns match."""
    parts = {split: load_processed_split(dataset, split) for split in SPLITS}
    X_train, Y_train = parts["training"]

    for split in ("validation", "test"):
        X, Y = parts[split]
        if X.columns.tolist() != X_train.columns.tolist():
            raise ValueError(
                f"Training and {split} CSVs must have the same feature columns "
                "in the same order."
            )
        if Y.columns.tolist() != Y_train.columns.tolist():
            raise ValueError(
                f"Training and {split} CSVs must have the same label columns "
                "in the same order."
            )

    return parts
