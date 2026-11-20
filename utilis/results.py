from pathlib import Path

import numpy as np
import pandas as pd

# Where results are kept. Every experiment has its own folder under results/, named for
# what it is (results/tabular_only/, results/text_only/, ...), holding one CSV per seed
# plus summary.md. The summary is a small markdown file so that git keeps it
# (.gitignore skips *.csv).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# One table is written to the summary for each of these result columns.
MEASURES = (("f1", "F1"), ("ap", "AP"), ("precision", "Precision"), ("recall", "Recall"),
            ("threshold", "Cutoff picked on validation"))


def new_results_dir(experiment, root=RESULTS_DIR):
    """Return the folder for one experiment's results: results/<experiment>/.

    Results are never overwritten. If that folder already holds files, the next
    free numbered folder is used instead: results/<experiment>_2/, then _3/, ...
    """
    base = Path(root) / experiment
    folder, n = base, 1
    while folder.exists() and any(p.is_file() for p in folder.iterdir()):
        n += 1
        folder = base.with_name(f"{base.name}_{n}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def markdown_table(df, digits=3):
    """A DataFrame as a markdown table, index included, decimals rounded."""
    cell = lambda v: f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in df.itertuples():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def summary_markdown(title, description, results):
    """The short summary of an experiment: averages first, then one table per measure.

    results has one row per model and event, with the columns that
    utilis.metrics.score_event produces (threshold, precision, recall, f1, ap)
    plus "model" and "label". description is a list of plain sentences.
    """
    models = list(dict.fromkeys(results.model))
    events = list(dict.fromkeys(results.label))

    headline = (results.groupby("model", sort=False)[["f1", "ap"]].mean()
                .rename(columns={"f1": "Average F1", "ap": "Mean AP"}).rename_axis("Model"))
    notes = [*description, "All numbers are test scores. Each event's yes/no cutoff was picked on validation."]
    parts = [f"# {title}", "", *[f"- {note}" for note in notes], "",
             "## Average over the events", "", markdown_table(headline)]

    for column, name in MEASURES:
        table = (results.pivot(index="label", columns="model", values=column)
                 .reindex(index=events, columns=models).rename_axis("Event"))
        if column != "threshold":
            table.loc["Average"] = table.mean()
        parts += ["", f"## {name} per event", "", markdown_table(table, digits=2 if column == "threshold" else 3)]

    return "\n".join(parts) + "\n"


def seeds_summary(title, description, seeds, seed_results, average=None):
    """summary.md for an experiment that trains one model per seed.

    seed_results holds each seed's per-event test scores, in the order of seeds.
    average holds the scores of the seed-averaged prediction, or None for one seed.
    """
    description = list(description)
    frames = [r.assign(model=f"Seed {seed}") for seed, r in zip(seeds, seed_results)]

    if average is not None:
        f1 = [r.f1.mean() for r in seed_results]
        ap = [r.ap.mean() for r in seed_results]
        description.append(
            f"Single models: average F1 {np.mean(f1):.3f} +/- {np.std(f1):.3f}, "
            f"mean AP {np.mean(ap):.3f} +/- {np.std(ap):.3f} (mean +/- SD over {len(seeds)} seeds)."
        )
        frames.insert(0, average)

    return summary_markdown(title, description, pd.concat(frames, ignore_index=True))
