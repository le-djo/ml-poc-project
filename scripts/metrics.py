"""
metrics.py — evaluation functions for the pump & dump anomaly detection project.

Primary metric   : Recall@LaMargia — fraction of true pump events flagged as anomalies.
Secondary metric : Anomaly score AUC — discriminative power of the continuous score.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def recall_at_lamorgia(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall on the La Morgia pump events (gt == 1).

    In regulatory surveillance, missing a real manipulation is worse than
    generating a false alarm — hence recall is the primary metric, not F1.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_pred : array-like of {0, 1}  (binarised anomaly predictions)

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pump_mask = y_true == 1
    if pump_mask.sum() == 0:
        return 0.0
    return float((y_pred[pump_mask] == 1).sum() / pump_mask.sum())


def anomaly_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC of the raw anomaly score against the binary pump label.

    Isolation Forest and LOF return negative scores (more negative = more anomalous).
    This function negates them so that higher score = more anomalous, aligning with
    the standard AUC convention where higher = more positive class.

    Parameters
    ----------
    y_true  : array-like of {0, 1}
    scores  : array-like of floats  (raw model decision_function output)

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    # Negate so that larger value = more anomalous (aligns with y_true=1 for pumps)
    return float(roc_auc_score(y_true, -scores))


def threshold_from_contamination(scores: np.ndarray, contamination: float) -> float:
    """
    Return the score threshold that marks the top `contamination` fraction
    of observations as anomalies.

    Parameters
    ----------
    scores        : raw anomaly scores (lower = more anomalous for IF/LOF)
    contamination : float in (0, 1)

    Returns
    -------
    float threshold — observations with score <= threshold are flagged
    """
    return float(np.percentile(scores, contamination * 100))


def evaluate(y_true: np.ndarray, scores: np.ndarray, contamination: float = 0.001) -> dict:
    """
    Full evaluation bundle.

    Parameters
    ----------
    y_true        : true binary labels
    scores        : raw anomaly scores from the model
    contamination : fraction of the dataset flagged as anomalies

    Returns
    -------
    dict with keys: recall, auc, n_flagged, n_true_pumps, contamination
    """
    threshold = threshold_from_contamination(scores, contamination)
    y_pred = (scores <= threshold).astype(int)
    return {
        "recall_at_lamorgia": recall_at_lamorgia(y_true, y_pred),
        "anomaly_auc": anomaly_auc(y_true, scores),
        "n_flagged": int(y_pred.sum()),
        "n_true_pumps": int(np.asarray(y_true).sum()),
        "contamination": contamination,
    }
