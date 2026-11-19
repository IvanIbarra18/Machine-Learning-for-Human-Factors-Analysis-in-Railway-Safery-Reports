import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Counts, hours, experience, and visibility from preprocessing_data.NUMERIC.
# The remaining C3RS columns are checkbox indicators, each a categorical feature.
NUMERIC_FIELDS = (
    "experience / total", "experience / in craft", "crew size", "workgroupsize",
    "hours into shift", "visibility / visibility", "total head end locomotives",
    "number of cars", "number in service",
)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("Training features must contain rows and columns.")
    numeric_columns = [c for c in X_train if any(k in c for k in NUMERIC_FIELDS)]
    categorical_columns = [c for c in X_train if c not in numeric_columns]
    if not numeric_columns or not categorical_columns:
        raise ValueError("Expected C3RS numeric fields and checkbox indicators.")
    if not X_train[categorical_columns].isin([0, 1]).all().all():
        raise ValueError("C3RS checkbox inputs must be 0 or 1. Regenerate processed CSVs.")
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric_columns),
        ("categorical", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        ), categorical_columns),
    ], remainder="drop")
