import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    if X_train.empty or X_train.shape[1] == 0:
        raise ValueError("Training features must contain rows and columns.")

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            # StandardScaler applies z = (x - mean) / standard deviation.
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False,),),
        ])

        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
