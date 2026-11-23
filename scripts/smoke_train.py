"""Tiny end-to-end smoke train on a subsample (used by CI).

Trains the TF-IDF pipeline on 20 examples per intent, evaluates on 200
held-out examples, and asserts the pipeline actually learns (accuracy well
above the 1/77 chance level). Finishes in seconds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from src.data import load_banking77
from src.train import train_tfidf_pipeline

TRAIN_PER_CLASS = 20
TEST_SIZE = 200


def main():
    dataset = load_banking77()
    texts = np.array(dataset["train"]["text"])
    labels = np.array(dataset["train"]["label"])

    rng = np.random.default_rng(0)
    keep = []
    for label in np.unique(labels):
        where = np.where(labels == label)[0]
        keep.extend(rng.choice(where, size=TRAIN_PER_CLASS, replace=False))
    pipeline = train_tfidf_pipeline(texts[keep], labels[keep],
                                    max_features=5000, seed=0)

    test_texts = dataset["test"]["text"][:TEST_SIZE]
    test_labels = np.array(dataset["test"]["label"][:TEST_SIZE])
    accuracy = accuracy_score(
        test_labels, pipeline["classifier"].predict(
            pipeline["extractor"].transform(test_texts)))
    print(f"smoke accuracy on {TEST_SIZE} test examples: {accuracy:.3f}")
    assert accuracy > 0.2, "pipeline did not learn above chance"
    print("smoke train OK")


if __name__ == "__main__":
    main()
