"""Evaluate the trained models on the BANKING77 test split.

Reproduce with ``make eval``. Every number quoted in the README comes from
the files this script writes into results/ - no hand-typed metrics.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.data import load_banking77
from src.evaluate import (
    abstention_curve,
    compute_metrics,
    confusion_dataframe,
    per_class_f1,
    top_confusions,
)

CACHE_DIR = Path("data/cache")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
ABSTENTION_THRESHOLDS = [round(t, 2) for t in np.arange(0.0, 1.0, 0.05)]


def test_features(name, extractor, test_texts):
    """Reuse cached test features when available; recompute otherwise."""
    cache = CACHE_DIR / f"test_{name}.npz"
    if name == "tfidf":
        return extractor.transform(test_texts)
    if cache.exists() and sum(len(c) for c in np.load(cache).values()) \
            == len(test_texts):
        return np.concatenate(list(np.load(cache).values()))
    return extractor.transform(test_texts)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_banking77()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    label_names = dataset["train"].features["label"].names

    tfidf_extractor = joblib.load(MODELS_DIR / "tfidf_extractor.joblib")
    feature_sets = {
        "tfidf": test_features("tfidf", tfidf_extractor, test_texts),
    }
    emb_cache = CACHE_DIR / "test_embeddings.npz"
    if emb_cache.exists():
        feature_sets["embeddings"] = np.concatenate(
            list(np.load(emb_cache).values()))
    classifiers = {
        "tfidf": joblib.load(MODELS_DIR / "tfidf_classifier.joblib"),
        "embeddings": joblib.load(MODELS_DIR / "embedding_classifier.joblib"),
    }

    metrics, curves, predictions = {}, [], {}
    for name, features in feature_sets.items():
        probabilities = classifiers[name].predict_proba(features)
        y_pred = np.argmax(probabilities, axis=1)
        predictions[name] = y_pred
        metrics[name] = compute_metrics(test_labels, y_pred)
        confusion_dataframe(test_labels, y_pred, label_names).to_csv(
            RESULTS_DIR / f"confusion_{name}.csv")
        per_class_f1(test_labels, y_pred, label_names).to_csv(
            RESULTS_DIR / f"per_class_f1_{name}.csv", index=False)
        with open(RESULTS_DIR / f"classification_report_{name}.txt", "w") as fh:
            fh.write(classification_report(test_labels, y_pred,
                                           target_names=label_names,
                                           zero_division=0))
        for row in abstention_curve(test_labels, probabilities,
                                    ABSTENTION_THRESHOLDS):
            curves.append({"model": name, **row})

    pd.DataFrame(curves).to_csv(RESULTS_DIR / "abstention.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(RESULTS_DIR / "error_analysis.md", "w", encoding="utf-8") as fh:
        for name, y_pred in predictions.items():
            fh.write(f"# Error analysis: top confusions ({name} model)\n\n")
            for item in top_confusions(test_labels, y_pred, test_texts,
                                       label_names, k=10):
                fh.write(f"## {item['true_intent']} -> "
                         f"{item['predicted_intent']} ({item['count']}x)\n\n")
                fh.writelines(f"- \"{example}\"\n" for example in item["examples"])
                fh.write("\n")

    print(json.dumps(metrics, indent=2))
    print("done -> results/")


if __name__ == "__main__":
    main()
