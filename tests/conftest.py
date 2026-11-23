import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import TfidfExtractor

TINY_CORPUS = {
    "card_lost": ["i lost my card", "my card is gone", "lost credit card",
                  "someone stole my card"],
    "top_up": ["how do i top up", "add money to my account",
               "put cash on my card", "top up my balance please"],
    "exchange_rate": ["what is the exchange rate",
                      "rate for euros today", "dollar exchange rate",
                      "what rate do you use"],
}


def build_tiny_model_dir(tmp_path):
    """Train a real (tiny) TF-IDF pipeline and save standard artifacts."""
    texts, labels, label_names = [], [], sorted(TINY_CORPUS)
    for name in label_names:
        texts.extend(TINY_CORPUS[name])
        labels.extend([label_names.index(name)] * len(TINY_CORPUS[name]))

    extractor = TfidfExtractor(max_features=500)
    features = extractor.fit_transform(texts)
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(features, np.array(labels))

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump(extractor, model_dir / "tfidf_extractor.joblib")
    joblib.dump(classifier, model_dir / "tfidf_classifier.joblib")
    with open(model_dir / "label_names.json", "w") as fh:
        json.dump(label_names, fh)
    with open(model_dir / "config.json", "w") as fh:
        json.dump({"dataset": "tiny-synthetic",
                   "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                   "embedding_frozen": True,
                   "default_abstain_threshold": 0.5}, fh)
    return model_dir, label_names


@pytest.fixture()
def tiny_model_dir(tmp_path):
    model_dir, _ = build_tiny_model_dir(tmp_path)
    return model_dir
