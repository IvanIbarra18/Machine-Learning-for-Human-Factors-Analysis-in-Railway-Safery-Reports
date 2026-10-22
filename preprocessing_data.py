# Add code to preprocess data

# Use this file as a single time preprocessing script

# Store proocessed dataset in datasets/processed

# Store processed csvs in the form of datasets/processed/ A_training.csv, A_validation.csv, A_test.csv

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = ROOT / "src"

# Load the .env file
ENV_FILE = NOTEBOOK_DIR / ".env" if (NOTEBOOK_DIR / ".env").is_file() else find_dotenv()
if not ENV_FILE:
    raise FileNotFoundError("No .env found in the repo root or src/.")
load_dotenv(ENV_FILE)

# The six event categories we want to predict.
# A report has the event if that column is True, and not if it is blank.
TARGETS = {
    "Station Ops":         "Events Anomaly / Operating Cab / Car Event - Station Operations",
    "Speed - MAS":         "Events Anomaly / Speed Deviation - Maximum Authorized Speed",
    "Emerg/Penalty Brake": "Events Anomaly / Braking Event - Emergency / Penalty Brake Application",
    "Mechanical (less)":   "Events Anomaly / Mechanical - Less Severe",
    "Signal - OBCS":       "Events Anomaly / Signal Events - On Board Control System",
    "Speed - TSR":         "Events Anomaly / Speed Deviation - Temporary Speed Restriction",
}

# Columns that are never used as model inputs (NEVER GIVEN TO THE MODEL): IDs, dates, free text, and anything that records the event itself or its outcome.
EXCLUDE_PREFIX = ("ACN", "Report Classification", "Time Date", "Person Narrative",
                  "Person Callback", "Person C3RS Report Number", "Person TrainReference",
                  "Component", "Events", "Assessments", "Synopsis")

# Train fields that directly restate one of our targets (data leakage), so they are removed too.
LEAKAGE = ("EmergencyBrakesApplied", "PenaltyBrakesApplied", "SpeedRestriction",
           "TypeOfTemporaryRestriction")

# Column groups we DO use as inputs.
INPUT_PREFIX = ("Time", "Environment", "Person", "Train")

# Experience is stored as ranges. We turn each range into a single number (the midpoint, or 35 for "31+").
EXP_BINS = {"0-2": 1, "3-5": 4, "6-10": 8, "11-20": 15, "21-30": 25, "31+": 35}

# Columns that hold numbers (counts, hours, years) instead of True/blank checkboxes.
NUMERIC = ("Experience / Total", "Experience / In Craft", "Crew Size", "WorkGroupSize",
           "Hours Into Shift", "Visibility / Visibility", "Total Head End Locomotives",
           "Number Of Cars", "Number In Service")

# Checkbox columns are True when selected and blank otherwise. 
def selected(x):
    return x is True or x == "True" or (isinstance(x, str) and x not in ("False", ""))

# Turns a column of mixed number formats into plain numbers.
# "3-5"  -> 4      (experience range)
# "31+"  -> 35     (experience range)
# "2:45" -> 2.75   (2 hours 45 minutes)
# "7"    -> 7      (normal number)
# blank  -> NaN    (missing)
def to_number(series):
    v = series.astype("string").str.strip()

    # Try 1: Is it an experience range? 
    # If so, replace it with the midpoint.
    ranges = v.map(EXP_BINS)

    # Try 2: Is it a time in hours:minutes?
    # If so, convert to decimal hours.
    hm = v.str.extract(r"^(\d+):(\d{2})$").astype(float)
    hours_minutes = hm[0] + hm[1] / 60

    # Try 3: Is it a plain number (with an optional + sign)?
    # If so, convert to float.
    plain = pd.to_numeric(v.str.replace("+", "", regex=False), errors="coerce")

    # Use the first successful conversion, in order: ranges, hours:minutes, plain number.
    return ranges.astype(float).fillna(hours_minutes).fillna(plain).astype("float64")


# Build the table the model learns from. 
# Each row is a report, each column is a feature. The label column is added later.
# Many fields repeat once per person in the report.
# We combine each group into a single column.
def build_features(df):

    # Keep ONLY input columns and drop excluded & leakage columns.
    cols = [c for c in df.columns if c.startswith(INPUT_PREFIX)
            and not c.startswith(EXCLUDE_PREFIX) and not any(k in c for k in LEAKAGE)]

    # Strip the trailing person number to get each field's base name.
    # Ex: "Person HumanFactors / Fatigue 2" -> "Person HumanFactors / Fatigue"
    base = {c: re.sub(r"\s+\d+$", "", c) for c in cols}

    feats = {}

    # Go through each field one time, combining the per-person copies into a single column.
    for b in dict.fromkeys(base.values()): 
        slot = [c for c in cols if base[c] == b]
        # Person 1's copy (who wrote the report).
        first = df[slot[0]]

        if any(k in b for k in NUMERIC):
            feats[b] = to_number(first)

            if "Visibility" in b:
                feats[b] = np.log1p(feats[b].clip(lower=0))
        else:
            # 1 if ANY person in the report has it selected, else 0.
            feats[b] = df[slot].map(selected).any(axis=1).astype(int)

    X = pd.DataFrame(feats)

    # Drop columns that have the same value in every report.
    return X.loc[:, X.nunique(dropna=True) > 1]


def main():
    #   --raw     the CSV file (default: DATA_PATH from .env)
    #   --target  which event category becomes the yes/no "label" column
    #   --out     where to save the three output files
    ap = argparse.ArgumentParser(description="Prepare the C3RS data for train.py")
    ap.add_argument("--raw", default=os.getenv("DATA_PATH"),
                    help="Raw CSV (default: DATA_PATH from .env, relative to src/)")
    ap.add_argument("--target", default="Station Ops", choices=list(TARGETS))
    ap.add_argument("--out", default="datasets/processed/A",
                    help="Output folder, relative to the repo root")
    a = ap.parse_args()
    if not a.raw:
        ap.error(f"DATA_PATH is not set in {ENV_FILE}. Add DATA_PATH=... or pass --raw.")

    raw = (NOTEBOOK_DIR / a.raw).resolve()
    out = ROOT / a.out
    if not raw.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw}  (DATA_PATH={a.raw!r}, read from {NOTEBOOK_DIR})")

    # Read CSV and clean up column names (remove extra spaces).
    print(f"Reading {raw}")
    df = pd.read_csv(raw, low_memory=False)
    df.columns = [" ".join(str(c).split()) for c in df.columns]

    # Build the input table.
    # Add the answer column "label": 1 = event happened, 0 = not.
    X = build_features(df)
    X.columns = [c.lower() for c in X.columns]
    X["label"] = df[TARGETS[a.target]].notna().astype(int)

    # Split into 70% training, 15% validation, 15% test.
    train, rest = train_test_split(X, test_size=0.30, stratify=X["label"], random_state=42)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=42)

    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("training", train), ("validation", val), ("test", test)]:
        part.to_csv(out / f"A_{name}.csv", index=False)
        print(f"{out / f'A_{name}.csv'}: {len(part):,} rows, "
              f"{part.shape[1] - 1} features, label=1 in {part['label'].mean():.1%}")
    print(f"Target: {a.target}")


if __name__ == "__main__":
    main()