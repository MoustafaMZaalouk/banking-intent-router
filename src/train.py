"""Training and artifact persistence for the intent router.

Only the scikit-learn classifiers (logistic regression) are trained here,
each from scratch on its feature matrix. The sentence-embedding model is
used frozen - nothing about it is trained or fine-tuned.
"""

import json
import time
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.features import EMBEDDING_MODEL_NAME, EmbeddingExtractor, TfidfExtractor

DEFAULT_MAX_FEATURES = 20000
DEFAULT_SEED = 42


def train_tfidf_pipeline(texts, labels, max_features=DEFAULT_MAX_FEATURES,
                         seed=DEFAULT_SEED):
    """Fit TF-IDF features plus a logistic-regression classifier."""
    start = time.perf_counter()
    extractor = TfidfExtractor(max_features=max_features)
    features = extractor.fit_transform(texts)
    feature_seconds = time.perf_counter() - start

    start = time.perf_counter()
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(features, labels)
    fit_seconds = time.perf_counter() - start

    return {
        "extractor": extractor,
        "classifier": classifier,
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
    }


def train_embedding_pipeline(texts, labels, model_name=EMBEDDING_MODEL_NAME,
                             seed=DEFAULT_SEED, extractor=None):
    """Fit a logistic-regression classifier on frozen embeddings."""
    if extractor is None:
        extractor = EmbeddingExtractor(model_name)

    start = time.perf_counter()
    features = extractor.transform(texts)
    feature_seconds = time.perf_counter() - start

    start = time.perf_counter()
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(features, labels)
    fit_seconds = time.perf_counter() - start

    return {
        "extractor": extractor,
        "classifier": classifier,
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
    }


def save_artifacts(out_dir, label_names, tfidf_pipeline, embedding_pipeline,
                   config):
    """Persist trained pipelines and metadata so the API can reload them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(tfidf_pipeline["extractor"], out_dir / "tfidf_extractor.joblib")
    joblib.dump(tfidf_pipeline["classifier"], out_dir / "tfidf_classifier.joblib")
    if embedding_pipeline is not None:
        joblib.dump(embedding_pipeline["classifier"],
                    out_dir / "embedding_classifier.joblib")

    with open(out_dir / "label_names.json", "w", encoding="utf-8") as handle:
        json.dump(list(label_names), handle, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
