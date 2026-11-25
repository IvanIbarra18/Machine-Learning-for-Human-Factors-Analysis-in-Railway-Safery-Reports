import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# Scoring shared by scripts/run_baselines.py and train.py, so every model is scored the same way.

# Cutoffs tried when turning a probability (0 to 1) into yes/no.
CUTOFFS = np.linspace(0.02, 0.98, 97)


def best_threshold(y, p):
    """Return the cutoff with the best F1. Use validation data only, never test."""
    return float(CUTOFFS[np.argmax([f1_score(y, p >= t, zero_division=0) for t in CUTOFFS])])


def score_event(y_val, p_val, y_test, p_test):
    """Pick the yes/no cutoff on validation, then score one event on the test set."""
    threshold = best_threshold(y_val, p_val)
    pred = p_test >= threshold
    return dict(
        threshold=threshold,
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        ap=average_precision_score(y_test, p_test),
    )


def average_precision_per_event(Y, P):
    """Average precision of each event (column). Y holds 0/1 answers, P probabilities."""
    Y = np.asarray(Y)
    P = np.asarray(P)

    if Y.shape != P.shape or Y.ndim != 2:
        raise ValueError("Y and P must be 2-D arrays of the same shape.")

    return [float(average_precision_score(Y[:, j], P[:, j])) for j in range(Y.shape[1])]
