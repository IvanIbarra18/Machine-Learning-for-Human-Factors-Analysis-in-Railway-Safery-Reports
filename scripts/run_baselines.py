# RQ1: To what extent can structured human-factor, environmental, personnel, and operational features classify the six
# recorded event categories in NASA C3RS reports?

# Trains 3 models (Logistic Regression, Random Forest, XGBoost) on the same data
# and saves their scores to results/results_per_label.csv.

# One model, quick test:
# python scripts/run_baselines.py --models LogReg
# python scripts/run_baselines.py --models RandomForest
# python scripts/run_baselines.py --models XGBoost

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from preprocessing_data import NOTEBOOK_DIR, ROOT, TARGETS, build_features

SEED = 42
MODELS = ("LogReg", "RandomForest", "XGBoost")

# Split the reports: 70% training, 15% validation, 15% test
def make_split(Y, narrative):
    # Give every unique narrative a group number.
    grp = narrative.factorize()[0]

    # Describe each report's events as a code.
    # Very rare codes are grouped as "rare" so the split doesn't fail on them.
    key = Y.astype(str).agg("".join, axis=1)
    key = key.where(key.map(key.value_counts()) >= 30, "rare")

    # Split the groups (NOT single reports).
    g = pd.DataFrame({"g": grp, "k": key}).drop_duplicates("g")
    g_tr, g_rest = train_test_split(g, test_size=0.30, stratify=g.k, random_state=SEED)
    g_va, g_te = train_test_split(g_rest, test_size=0.50, stratify=g_rest.k, random_state=SEED)

    # Turn the chosen groups back into row numbers.
    rows = lambda part: np.where(np.isin(grp, part.g.values))[0]
    return rows(g_tr), rows(g_va), rows(g_te)


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


# Models give a probability (0 to 1). We need a cutoff to turn it into yes/no.
def best_threshold(y, p):
    cutoffs = np.linspace(0.02, 0.98, 97)
    return float(cutoffs[np.argmax([f1_score(y, p >= t, zero_division=0) for t in cutoffs])])


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

        # 2. Pick the yes/no cutoff on validation.
        thr = best_threshold(y[va], p_val)

        # 3. Score 1x on the test set (which the model has never seen).
        p_test = m.predict_proba(X[te])[:, 1]
        pred = p_test >= thr
        rows.append(dict(setting="Structured", model=name, label=label, params=json.dumps(p),
                         threshold=thr,
                         precision=precision_score(y[te], pred, zero_division=0),
                         recall=recall_score(y[te], pred, zero_division=0),
                         f1=f1_score(y[te], pred, zero_division=0),
                         ap=average_precision_score(y[te], p_test),
                         n_test=len(te)))
    return pd.DataFrame(rows)


def main():
    # Choose which models to run.
    ap = argparse.ArgumentParser(description="RQ1: compare 3 models on the coded fields")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    a = ap.parse_args()

    # Read the CSV and clean up column names.
    raw = (NOTEBOOK_DIR / os.getenv("DATA_PATH")).resolve()
    print(f"Reading {raw}")
    df = pd.read_csv(raw, low_memory=False)
    df.columns = [" ".join(str(c).split()) for c in df.columns]

    Y = pd.DataFrame({k: df[v].notna().astype(int) for k, v in TARGETS.items()})

    narr_cols = [c for c in df.columns if c.startswith("Person Narrative / ")]
    narrative = (df[narr_cols].astype("string").fillna("").apply(lambda s: s.str.strip()).agg(" ".join, axis=1).str.strip())

    X = build_features(df)
    tr, va, te = make_split(Y, narrative)

    # Fill missing numbers with the median value from the training part,
    # then scale every column to between -1 and 1.
    X = X.fillna(X.iloc[tr].median())
    X = MaxAbsScaler().fit(X.iloc[tr]).transform(X)
    print(f"{len(df):,} reports, {X.shape[1]} features, " f"split {len(tr):,} / {len(va):,} / {len(te):,}")

    # Train and score each model.
    results = []
    for name in a.models:
        start = time.time()
        r = evaluate(name, Y, X, tr, va, te)
        results.append(r)
        print(f"{name:12s} average F1 {r.f1.mean():.3f}   mean AP {r.ap.mean():.3f}   " f"({time.time() - start:.0f}s)")

    out = ROOT / "results" / "results_per_label.csv"
    out.parent.mkdir(exist_ok=True)
    pd.concat(results, ignore_index=True).to_csv(out, index=False)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()