"""Benchmark and validate the quantized ONNX embedder vs PyTorch MiniLM.

Checks that ONNX embeddings stay faithful (cosine similarity ~1), re-scores
the test set with the same frozen classifier, and measures latency. Writes
results/latency_onnx.json - the only ONNX numbers the README may quote.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from src.data import load_banking77
from src.predict import IntentRouter

RESULTS_DIR = Path("results")
ONNX_MODEL = Path("models/embedder_onnx/model_quantized.onnx")
N_REQUESTS = 200


def build_onnx_encoder():
    session = ort.InferenceSession(str(ONNX_MODEL))
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2")

    def encode(texts):
        encoded = tokenizer(list(texts), padding=True, truncation=True,
                            max_length=512, return_tensors="np")
        outputs = session.run(None, dict(encoded))
        embeddings = outputs[0]
        if embeddings.ndim == 3:  # last_hidden_state -> mean pool
            mask = encoded["attention_mask"][..., None]
            embeddings = (embeddings * mask).sum(axis=1) / \
                np.maximum(mask.sum(axis=1), 1)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norm, 1e-12)

    return encode


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_banking77()
    test_texts = dataset["test"]["text"]
    test_labels = np.array(dataset["test"]["label"])
    router = IntentRouter("models")
    onnx_encode = build_onnx_encoder()

    sample = test_texts[:64]
    reference = router._get_embedder().transform(sample)
    cosine = float(np.mean(np.sum(reference * onnx_encode(sample), axis=1)))

    classifier = joblib.load("models/embedding_classifier.joblib")
    batches = [onnx_encode(test_texts[i:i + 256])
               for i in range(0, len(test_texts), 256)]
    onnx_accuracy = float(accuracy_score(
        test_labels, classifier.predict(np.concatenate(batches))))

    latencies = []
    start_all = time.perf_counter()
    for text in test_texts[:N_REQUESTS]:
        start = time.perf_counter()
        onnx_encode([text])
        latencies.append((time.perf_counter() - start) * 1000)
    total = time.perf_counter() - start_all
    latencies = np.array(latencies)

    report = {
        "model": str(ONNX_MODEL),
        "cosine_similarity_vs_pytorch": round(cosine, 5),
        "test_accuracy_with_onnx_embeddings": round(onnx_accuracy, 4),
        "n_requests": N_REQUESTS,
        "mean_ms": round(float(latencies.mean()), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_per_s": round(len(latencies) / total, 1),
        "note": "dynamically quantized int8 ONNX, single request at a time; "
                "classifier is the same frozen-feature logistic regression",
    }
    with open(RESULTS_DIR / "latency_onnx.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
