import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "src"
SITE_DIR = ROOT / "website"
RESULTS_CSV = ROOT / "results" / "results_per_label.csv"
PROCESSED_DIR = ROOT / "datasets" / "processed" / "A"

# Load the .env file.
ENV_FILE = NOTEBOOK_DIR / ".env" if (NOTEBOOK_DIR / ".env").is_file() else find_dotenv()
if not ENV_FILE:
    raise FileNotFoundError("No .env found in the repo root or src/.")
load_dotenv(ENV_FILE)
if not os.getenv("DATA_PATH"):
    raise ValueError(f"DATA_PATH is not set in {ENV_FILE}.")
RAW = (NOTEBOOK_DIR / os.getenv("DATA_PATH")).resolve()

# The 6 event types, with the names shown on the website.
TARGETS = {
    "Station operations":           "Events Anomaly / Operating Cab / Car Event - Station Operations",
    "Speed: max authorized":        "Events Anomaly / Speed Deviation - Maximum Authorized Speed",
    "Emergency / penalty brake":    "Events Anomaly / Braking Event - Emergency / Penalty Brake Application",
    "Mechanical, less severe":      "Events Anomaly / Mechanical - Less Severe",
    "Signal: on-board control":     "Events Anomaly / Signal Events - On Board Control System",
    "Speed: temporary restriction": "Events Anomaly / Speed Deviation - Temporary Speed Restriction",
}

RESULT_LABELS = {
    "Station Ops": "Station operations", "Speed - MAS": "Speed: max authorized",
    "Emerg/Penalty Brake": "Emergency / penalty brake", "Mechanical (less)": "Mechanical, less severe",
    "Signal - OBCS": "Signal: on-board control", "Speed - TSR": "Speed: temporary restriction",
}

# Our first target, mentioned on the website to explain why we switched.
SAFETY_CONCERN = "Events Anomaly / Safety Concern"

COUNT_WORDS = ["None of the six", "One", "Two", "Three", "Four", "Five", "Six"]

NARRATIVE_PLACEHOLDER = re.compile(
    r"\[?(?:(?:second|third) person|person \d+)(?: reference)? added to supplement the records?\.?\]?\.?", re.I)

# Find the per-person columns of one kind, in order.
def slots(df, kind):
    pat = re.compile(rf"Person {kind} / {kind} (\d+)")
    return sorted((c for c in df.columns if pat.fullmatch(c)), key=lambda c: int(pat.fullmatch(c).group(1)))


# Get those columns as text w/ blanks as "" and extra spaces removed.
def text_frame(df, kind):
    return df[slots(df, kind)].astype("string").fillna("").apply(lambda s: s.str.strip())


# Count everything the website shows about the data itself.
def data_stats(df):
    n = len(df)
    month = pd.to_datetime(df["Time Date"].astype(str).str.strip(), format="%Y-%m", errors="coerce")

    # Reports per year.
    # If the last year doesn't have all 12 months yet,
    # mark it as "partial" so the website draws it striped.
    years = month.dt.year.value_counts().sort_index()
    last_year = int(years.index.max())
    last_months = month[month.dt.year == last_year].dt.month
    per_year = [{"year": int(y), "reports": int(v),"partial": bool(y == last_year and last_months.nunique() < 12)} for y, v in years.items()]
    partial_note = (f"{last_year} covers {month[month.dt.year == last_year].min():%B}–" f"{month[month.dt.year == last_year].max():%B} only") if per_year[-1]["partial"] else ""

    # Report text.
    narr = text_frame(df, "Narrative")
    narr_ok = narr.ne("") & ~narr.apply(lambda s: s.str.fullmatch(NARRATIVE_PLACEHOLDER, na=False))
    cb = text_frame(df, "Callback")
    cb_ok = cb.ne("") & ~cb.apply(lambda s: s.str.lower().str.startswith("unable to contact"))

    # How many reports have both a real narrative and a real callback.
    callback_reports = int((narr_ok.any(axis=1) & cb_ok.any(axis=1)).sum())

    # Number of words in each report's narratives.
    narrative_words = narr.where(narr_ok, "").agg(" ".join, axis=1).str.split().str.len()

    # The 6 event types as 0/1 columns, and how many of them each report has.
    Y = pd.DataFrame({name: df[col].notna().astype(int) for name, col in TARGETS.items()})
    per_report = Y.sum(axis=1).value_counts().sort_index()

    # Human factors: count reports where ANY person has each factor checked.
    # Skip "Other" because it's not a real factor name.
    hf_cols = [c for c in df.columns if c.startswith("Person HumanFactors /")]
    hf_base = {c: re.sub(r"\s+\d+$", "", c).split(" / ", 1)[1] for c in hf_cols}
    hf = []

    for name in dict.fromkeys(hf_base.values()):
        if name.lower() == "other":
            continue
        cols = [c for c in hf_cols if hf_base[c] == name]
        k = int(df[cols].map(lambda x: x is True or x == "True").any(axis=1).sum())
        hf.append({"name": name.replace("Human - Machine", "Human–machine").capitalize().replace("human–machine", "Human–machine"), "reports": k})

    # Most common first.
    hf.sort(key=lambda d: -d["reports"])

    # Everything the website needs about the data.
    return {
        "reports": n,
        "columns": int(df.shape[1]),
        "first_month": f"{month.min():%B %Y}",
        "last_month": f"{month.max():%B %Y}",
        "median_narrative_words": int(narrative_words[narrative_words > 0].median()),
        "callback_reports": callback_reports,
        "per_year": per_year,
        "partial_year_note": partial_note,
        "safety_concern": int(df[SAFETY_CONCERN].notna().sum()) if SAFETY_CONCERN in df else None,
        "targets": [{"name": k, "reports": int(v)} for k, v in Y.sum().items()],
        "per_report": [{"name": COUNT_WORDS[int(k)], "reports": int(v)} for k, v in per_report.items()],
        "human_factors": hf[:10],   # top 10
    }

# How many reports are in each file from preprocessing_data.py.
def split_stats():
    sizes = {}
    for part in ("training", "validation", "test"):
        f = PROCESSED_DIR / f"A_{part}.csv"
        if f.is_file():
            with open(f, encoding="utf-8") as fh:
                sizes[part] = sum(1 for _ in fh) - 1
    return sizes or None

MODEL_NAMES = {"LogReg": "Logistic Regression", "RandomForest": "Random Forest", "XGBoost": "XGBoost"}

def _order(rows):
    order = list(TARGETS)
    return sorted(rows, key=lambda d: order.index(d["name"]) if d["name"] in order else 99)


# RQ1 results: every model trained on the coded fields only ("Structured").
def rq1_stats(r):
    s = r[r["setting"] == "Structured"]
    if s.empty:
        return None
    models = [m for m in MODEL_NAMES if m in set(s["model"])]

    # For each model: its average scores across the 6 event types,
    # + every score for each event type (for the "Results by model" tables).
    summary = []
    for m in models:
        sm = s[s.model == m]
        summary.append({"key": m, "model": MODEL_NAMES[m],
                        "macro_f1": round(float(sm["f1"].mean()), 3),
                        "map": round(float(sm["ap"].mean()), 3),
                        "precision": round(float(sm["precision"].mean()), 3),
                        "recall": round(float(sm["recall"].mean()), 3),
                        "per_category": _order([{"name": RESULT_LABELS.get(row.label, row.label),
                                                 "precision": round(float(row.precision), 3),
                                                 "recall": round(float(row.recall), 3),
                                                 "f1": round(float(row.f1), 3),
                                                 "ap": round(float(row.ap), 3)}
                                                for row in sm.itertuples()])})

    # For each event type: every model's F1 side by side (for the dot chart).
    per_cat = []
    for label, g in s.groupby("label"):
        row = {"name": RESULT_LABELS.get(label, label)}
        row.update({m: round(float(g[g.model == m]["f1"].iloc[0]), 3) for m in models if (g.model == m).any()})
        per_cat.append(row)

    # Rank the models by average F1.
    ranked = sorted(summary, key=lambda d: -d["macro_f1"])
    best = ranked[0]
    clear_winner = len(ranked) > 1 and best["macro_f1"] - ranked[1]["macro_f1"] >= 0.02
    return {"models": summary, "per_category": _order(per_cat), "best_model": best["model"],
            "best_f1": best["macro_f1"], "best_map": best["map"], "clear_winner": bool(clear_winner),
            "test_reports": int(s["n_test"].iloc[0]) if "n_test" in s else None}

# RQ2 results: coded fields only vs. coded fields + narrative, for one model (XGBoost if there).
# Returns nothing until RQ2 has been run, so the website hides that section for now.
def rq2_stats(r):
    b_all = r[r["setting"] == "Structured + Narrative"]
    if b_all.empty:
        return None
    model = "XGBoost" if "XGBoost" in set(b_all["model"]) else b_all["model"].iloc[0]
    a = r[(r.setting == "Structured") & (r.model == model)].set_index("label")["f1"]
    b = b_all[b_all.model == model].set_index("label")["f1"]
    if a.empty:
        return None
    rows = [{"name": RESULT_LABELS.get(k, k), "structured": round(float(a[k]), 3),
             "narrative": round(float(b[k]), 3)} for k in a.index if k in b.index]
    return {"model": MODEL_NAMES.get(model, model), "macro_structured": round(float(a.mean()), 2),
            "macro_narrative": round(float(b.mean()), 2), "per_category": _order(rows)}


# Read the model results (from run_baselines.py) and get the RQ1 and RQ2 numbers.
# If there's no results file yet, the website shows "results coming soon".
def result_stats():
    if not RESULTS_CSV.is_file():
        return None
    r = pd.read_csv(RESULTS_CSV)
    out = {"rq1": rq1_stats(r), "rq2": rq2_stats(r)}
    return out if any(out.values()) else None

# Only run this part when you run the file directly.
if __name__ == "__main__":
    # Read the CSV and clean up extra spaces in the column names.
    print(f"Reading {RAW}")
    df = pd.read_csv(RAW, low_memory=False)
    df.columns = [" ".join(str(c).split()) for c in df.columns]

    # Put everything together: today's date, data counts, split sizes, model results,
    # and the milestones from progress.json.
    progress = json.loads((SITE_DIR / "progress.json").read_text(encoding="utf-8"))
    slides = progress.get("slides")
    if slides:
        slides["available"] = (SITE_DIR / slides.get("file", "")).is_file()
        
    site = {
        "data": data_stats(df),
        "splits": split_stats(),
        "results": result_stats(),
        "progress": progress,
    }

    out = SITE_DIR / "site_data.js"
    out.write_text("// Generated by build_site_data.py. Do not edit by hand.\nwindow.SITE_DATA = "
                   + json.dumps(site, indent=1, ensure_ascii=False) + ";\n", encoding="utf-8")
    
    print(f"Wrote {out}")
    print(f"  {site['data']['reports']:,} reports, {len(site['data']['per_year'])} years, "
          f"results: {', '.join(k.upper() for k, v in (site['results'] or {}).items() if v) or 'none found'}, "
          f"splits: {site['splits'] or 'none found'}")