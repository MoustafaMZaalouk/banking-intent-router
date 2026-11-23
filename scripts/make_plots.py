"""Regenerate every plot in plots/ from the CSV/JSON files in results/.

Reproduce with:  python scripts/make_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def plot_confusion(name):
    matrix = pd.read_csv(RESULTS_DIR / f"confusion_{name}.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix.values, cmap="viridis")
    ax.set_title(f"BANKING77 confusion matrix ({name})")
    ax.set_xlabel("predicted intent")
    ax.set_ylabel("true intent")
    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def plot_per_class_f1():
    tfidf = pd.read_csv(RESULTS_DIR / "per_class_f1_tfidf.csv")
    embeddings = pd.read_csv(RESULTS_DIR / "per_class_f1_embeddings.csv")
    merged = tfidf[["intent", "f1"]].merge(
        embeddings[["intent", "f1"]], on="intent",
        suffixes=("_tfidf", "_embeddings")).sort_values("f1_embeddings")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(merged)), merged["f1_tfidf"], ".", alpha=0.6,
            label="TF-IDF")
    ax.plot(range(len(merged)), merged["f1_embeddings"], ".", alpha=0.6,
            label="embeddings (frozen MiniLM)")
    ax.set_title("Per-intent F1 on BANKING77 test set (sorted by embedding F1)")
    ax.set_xlabel("intent (sorted)")
    ax.set_ylabel("F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)


def plot_abstention():
    curves = pd.read_csv(RESULTS_DIR / "abstention.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in curves.groupby("model"):
        group = group.dropna(subset=["accuracy"])
        ax.plot(group["coverage"], group["accuracy"], ".-", label=name)
    ax.set_title("Abstention trade-off: accuracy on answered queries")
    ax.set_xlabel("coverage (share of queries answered)")
    ax.set_ylabel("accuracy on answered queries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "abstention_curve.png", dpi=150)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_confusion("tfidf")
    plot_confusion("embeddings")
    plot_per_class_f1()
    plot_abstention()
    with open(RESULTS_DIR / "metrics.json") as fh:
        print(json.dumps(json.load(fh), indent=2))
    print("plots -> plots/")


if __name__ == "__main__":
    main()
