"""Measure single-request latency and throughput of each served model.

Reproduce with ``make benchmark``. Numbers land in results/latency.json and
are the only latency figures the README may quote.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data import load_banking77
from src.predict import IntentRouter

RESULTS_DIR = Path("results")
N_REQUESTS = 200


def benchmark(router, texts, model):
    latencies = []
    start_all = time.perf_counter()
    for text in texts:
        start = time.perf_counter()
        router.predict(text, model=model, abstain_threshold=0.5)
        latencies.append((time.perf_counter() - start) * 1000)
    total = time.perf_counter() - start_all
    latencies = np.array(latencies)
    return {
        "requests": len(texts),
        "mean_ms": round(float(latencies.mean()), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_per_s": round(len(texts) / total, 1),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_banking77()
    texts = dataset["test"]["text"][:N_REQUESTS]
    router = IntentRouter("models")

    report = {"n_requests": N_REQUESTS, "device": "CPU",
              "models": {}}
    for model in ("tfidf", "embeddings"):
        router.predict("warm up", model=model)
        report["models"][model] = benchmark(router, texts, model)
    report["note"] = ("single process, single request at a time; the "
                      "embedding path includes frozen MiniLM encoding")
    with open(RESULTS_DIR / "latency.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
