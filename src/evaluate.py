"""Evaluation helpers: metrics, confusion analysis, and abstention.

The numbers these functions compute are the honest headline results of the
project, so the tests in ``tests/test_evaluate.py`` check them against
hand-computed expectations, not just shapes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred):
    """Accuracy and F1 scores for one model's predictions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def confusion_dataframe(y_true, y_pred, label_names):
    """Confusion matrix as a labelled DataFrame (rows=true, cols=pred)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    return pd.DataFrame(matrix, index=label_names, columns=label_names)


def per_class_f1(y_true, y_pred, label_names):
    """Per-intent precision / recall / F1 / support as a DataFrame."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))),
        zero_division=0,
    )
    return pd.DataFrame({
        "intent": label_names,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def decide(probabilities, threshold):
    """Turn a class-probability vector into a routing decision.

    Returns (best_index, confidence, abstained). The router abstains when
    the top class probability is below ``threshold`` - better to hand the
    query to a human than to route it confidently to the wrong intent.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    return best_index, confidence, confidence < threshold


def abstention_curve(y_true, probabilities, thresholds):
    """Coverage/accuracy trade-off as the abstention threshold grows.

    For each threshold, ``coverage`` is the share of examples the router
    still answers and ``accuracy`` is measured only on those answered
    examples. A useful router should get more accurate as coverage drops.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    rows = []
    for threshold in thresholds:
        answered = confidences >= threshold
        coverage = float(np.mean(answered))
        if answered.any():
            accuracy = float(accuracy_score(y_true[answered],
                                            predictions[answered]))
        else:
            accuracy = None
        rows.append({
            "threshold": float(threshold),
            "coverage": coverage,
            "accuracy": accuracy,
            "abstained": int((~answered).sum()),
        })
    return rows


def top_confusions(y_true, y_pred, texts, label_names, k=10, examples_per_pair=2):
    """Most frequent (true, predicted) mistakes with example queries."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = y_true != y_pred

    pairs = {}
    for index in np.where(wrong)[0]:
        key = (int(y_true[index]), int(y_pred[index]))
        pairs.setdefault(key, []).append(int(index))

    ranked = sorted(pairs.items(), key=lambda item: len(item[1]), reverse=True)
    results = []
    for (true_index, pred_index), rows in ranked[:k]:
        results.append({
            "true_intent": label_names[true_index],
            "predicted_intent": label_names[pred_index],
            "count": len(rows),
            "examples": [texts[i] for i in rows[:examples_per_pair]],
        })
    return results
