from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "datasets" / "processed"
AVAILABLE_DATASETS = ("A",)
TARGET_COLUMN = "label"
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
) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = get_processed_dataset_path(dataset, split)
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names in {csv_path}: {duplicates}.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} is missing from {csv_path}."
        )
    if df.empty:
        raise ValueError(f"Processed dataset CSV is empty: {csv_path}.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Missing target labels found in {csv_path}.")

    y = df.pop(TARGET_COLUMN).astype(str)
    X = df
    if X.shape[1] == 0:
        raise ValueError(f"No feature columns found in {csv_path}.")

    return X, y


def load_train_validation(
    dataset: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train = load_processed_split(dataset, "training")
    X_validation, y_validation = load_processed_split(dataset, "validation")

    if X_train.columns.tolist() != X_validation.columns.tolist():
        raise ValueError(
            "Training and validation CSVs must have the same feature columns "
            "in the same order."
        )

    return X_train, y_train, X_validation, y_validation


def encode_labels(labels: pd.Series, class_names: list[str]) -> np.ndarray:
    label_to_id = {name: index for index, name in enumerate(class_names)}
    encoded = labels.map(label_to_id)

    if encoded.isna().any():
        unexpected = labels[encoded.isna()].unique().tolist()
        raise ValueError(f"Labels outside the known classes: {unexpected}")

    return encoded.to_numpy(dtype=np.int64)
