"""Loading helpers for the BANKING77 dataset.

BANKING77 (Casanueva et al., 2020) contains 13,083 customer-service
queries labelled with one of 77 banking intents. It is distributed under
CC BY 4.0 at https://huggingface.co/datasets/PolyAI/banking77.
"""

from pathlib import Path

from datasets import load_dataset, load_from_disk

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "banking77"
DATASET_NAME = "PolyAI/banking77"


def load_banking77():
    """Load BANKING77, downloading it once into ``data/banking77``."""
    if DATA_DIR.exists():
        return load_from_disk(str(DATA_DIR))
    dataset = load_dataset(DATASET_NAME)
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_DIR))
    return dataset


def split_texts_labels(dataset, split):
    """Return (texts, numeric_labels, label_names) for one split."""
    texts = dataset[split]["text"]
    labels = dataset[split]["label"]
    label_names = dataset[split].features["label"].names
    return texts, labels, label_names
