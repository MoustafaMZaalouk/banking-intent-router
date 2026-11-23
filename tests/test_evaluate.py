"""Metric correctness, checked against hand-computed expectations."""

import numpy as np
import pytest

from src.evaluate import (
    abstention_curve,
    compute_metrics,
    confusion_dataframe,
    decide,
    top_confusions,
)


def test_compute_metrics_hand_computed():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.75)
    # class 0: precision 1/1, recall 1/2 -> f1 2/3; class 1: 2/3, 1 -> 4/5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_confusion_dataframe_counts():
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 2]
    frame = confusion_dataframe(y_true, y_pred, ["a", "b", "c"])
    assert frame.loc["a", "b"] == 1
    assert frame.loc["b", "b"] == 2
    assert frame.values.sum() == 5


def test_decide_abstains_below_threshold():
    index, confidence, abstained = decide([0.6, 0.4], threshold=0.9)
    assert (index, abstained) == (0, True)
    assert confidence == pytest.approx(0.6)
    _, _, abstained = decide([0.6, 0.4], threshold=0.5)
    assert not abstained


def test_abstention_curve_tradeoff():
    """Accuracy on answered queries rises as coverage drops (crafted case)."""
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([
        [0.95, 0.05],  # correct, confident
        [0.10, 0.90],  # correct, confident
        [0.55, 0.45],  # correct, unsure
        [0.51, 0.49],  # wrong, unsure
    ])
    curve = abstention_curve(y_true, probabilities, [0.0, 0.6, 0.9])
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[0]["accuracy"] == pytest.approx(0.75)
    assert curve[1]["coverage"] == pytest.approx(0.5)
    assert curve[1]["accuracy"] == pytest.approx(1.0)
    assert curve[2]["abstained"] == 2


def test_top_confusions_orders_by_count():
    y_true = [0, 0, 0, 1]
    y_pred = [1, 1, 0, 0]
    texts = ["t1", "t2", "t3", "t4"]
    results = top_confusions(y_true, y_pred, texts, ["a", "b"], k=2)
    assert results[0]["true_intent"] == "a"
    assert results[0]["predicted_intent"] == "b"
    assert results[0]["count"] == 2
    assert results[0]["examples"] == ["t1", "t2"]
