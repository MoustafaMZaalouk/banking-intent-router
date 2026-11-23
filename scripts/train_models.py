"""Train both intent classifiers and save artifacts to models/.

Reproduce with ``make train``. The script checkpoints every expensive step
(features, embeddings, fits) under data/cache/ and models/, so re-running
it resumes where the previous run stopped. Only the logistic-regression
classifiers are trained; the sentence-embedding model stays frozen.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from src.data import DATASET_NAME, load_banking77
from src.features import EMBEDDING_MODEL_NAME, EmbeddingExtractor, TfidfExtractor
from src.train import DEFAULT_MAX_FEATURES, DEFAULT_SEED

CACHE_DIR = Path("data/cache")
MODELS_DIR = Path("models")
EMBED_BATCH = 500


def timed(step_times, name, func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    step_times[name] = step_times.get(name, 0.0) + round(
        time.perf_counter() - start, 2)
    return result


def encode_in_chunks(extractor, texts, cache_path, batch=EMBED_BATCH):
    """Encode texts, saving after every chunk so timeouts lose nothing."""
    done = 0
    chunks = []
    if cache_path.exists():
        chunks = list(np.load(cache_path).values())
        done = sum(len(chunk) for chunk in chunks)
    for start in range(done, len(texts), batch):
        chunk = extractor.transform(texts[start:start + batch])
        chunks.append(chunk)
        np.savez(cache_path, *chunks)
        print(f"  encoded {start + len(chunk)}/{len(texts)}", flush=True)
    return np.concatenate(chunks)


def main():
    started = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    step_times = {}

    dataset = load_banking77()
    train_texts = dataset["train"]["text"]
    train_labels = np.array(dataset["train"]["label"])
    test_texts = dataset["test"]["text"]
    label_names = dataset["train"].features["label"].names
    print(f"train={len(train_texts)} test={len(test_texts)} "
          f"intents={len(label_names)}", flush=True)

    tfidf_path = MODELS_DIR / "tfidf_extractor.joblib"
    train_x_path = CACHE_DIR / "train_tfidf.npz"
    if tfidf_path.exists() and train_x_path.exists():
        tfidf_extractor = joblib.load(tfidf_path)
        train_x = sparse.load_npz(train_x_path)
        print("tfidf features: loaded from cache", flush=True)
    else:
        tfidf_extractor = TfidfExtractor(max_features=DEFAULT_MAX_FEATURES)
        train_x = timed(step_times, "tfidf_feature_seconds",
                        tfidf_extractor.fit_transform, train_texts)
        joblib.dump(tfidf_extractor, tfidf_path)
        sparse.save_npz(train_x_path, train_x)
        print(f"tfidf features: {train_x.shape}", flush=True)

    tfidf_clf_path = MODELS_DIR / "tfidf_classifier.joblib"
    if tfidf_clf_path.exists():
        print("tfidf classifier: loaded from cache", flush=True)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=DEFAULT_SEED)
        timed(step_times, "tfidf_fit_seconds", clf.fit, train_x, train_labels)
        joblib.dump(clf, tfidf_clf_path)
        print("tfidf classifier: fitted", flush=True)

    train_emb_path = CACHE_DIR / "train_embeddings.npz"
    test_emb_path = CACHE_DIR / "test_embeddings.npz"
    if train_emb_path.exists() and len(np.load(train_emb_path)) > 0 \
            and sum(len(c) for c in np.load(train_emb_path).values()) \
            == len(train_texts):
        print("train embeddings: cached", flush=True)
    else:
        embedder = EmbeddingExtractor(EMBEDDING_MODEL_NAME)
        timed(step_times, "embedding_encode_train_seconds",
              encode_in_chunks, embedder, train_texts, train_emb_path)
    if test_emb_path.exists() and sum(
            len(c) for c in np.load(test_emb_path).values()) == len(test_texts):
        print("test embeddings: cached", flush=True)
    else:
        embedder = EmbeddingExtractor(EMBEDDING_MODEL_NAME)
        timed(step_times, "embedding_encode_test_seconds",
              encode_in_chunks, embedder, test_texts, test_emb_path)

    emb_clf_path = MODELS_DIR / "embedding_classifier.joblib"
    if emb_clf_path.exists():
        print("embedding classifier: loaded from cache", flush=True)
    else:
        train_embeddings = np.concatenate(list(np.load(train_emb_path).values()))
        clf = LogisticRegression(max_iter=1000, random_state=DEFAULT_SEED)
        timed(step_times, "embedding_fit_seconds", clf.fit,
              train_embeddings, train_labels)
        joblib.dump(clf, emb_clf_path)
        print("embedding classifier: fitted", flush=True)

    config = {
        "dataset": DATASET_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_frozen": True,
        "classifier": "sklearn.linear_model.LogisticRegression",
        "seed": DEFAULT_SEED,
        "default_abstain_threshold": 0.5,
    }
    label_path = MODELS_DIR / "label_names.json"
    config_path = MODELS_DIR / "config.json"
    if not label_path.exists():
        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(list(label_names), handle, indent=2)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    Path("results").mkdir(exist_ok=True)
    runtime_path = Path("results/runtime_train.json")
    if runtime_path.exists():
        with open(runtime_path) as handle:
            previous = json.load(handle)
        for key, value in previous.items():
            if key.endswith("_seconds") and key not in step_times:
                step_times[key] = value
    step_times["total_seconds"] = round(time.perf_counter() - started, 2)
    step_times["hardware"] = (f"{platform.machine()} CPU, "
                              f"{platform.processor() or 'unknown'}")
    step_times["python"] = platform.python_version()
    step_times["note"] = ("wall-clock on this machine; embedding weights are "
                          "downloaded once and frozen, never trained")
    with open(runtime_path, "w") as handle:
        json.dump(step_times, handle, indent=2)
    print("done -> models/", flush=True)


if __name__ == "__main__":
    main()
