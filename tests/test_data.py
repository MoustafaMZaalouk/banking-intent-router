"""Dataset validation: schema, label set, and train/test hygiene."""

import pytest

from src.data import load_banking77

EXPECTED_N_LABELS = 77
EXPECTED_TRAIN_SIZE = 10003
EXPECTED_TEST_SIZE = 3080


@pytest.fixture(scope="module")
def dataset():
    return load_banking77()


def test_schema(dataset):
    for split in ("train", "test"):
        assert dataset[split].column_names == ["text", "label"]
        assert dataset[split].features["text"].dtype == "string"
        assert dataset[split].features["label"].dtype == "int64"


def test_sizes_and_label_set(dataset):
    assert len(dataset["train"]) == EXPECTED_TRAIN_SIZE
    assert len(dataset["test"]) == EXPECTED_TEST_SIZE
    label_names = dataset["train"].features["label"].names
    assert len(label_names) == EXPECTED_N_LABELS
    assert set(dataset["train"]["label"]) == set(range(EXPECTED_N_LABELS))


def test_no_empty_texts(dataset):
    for split in ("train", "test"):
        assert all(text.strip() for text in dataset[split]["text"])


# BANKING77 has no ID column, so we check exact normalized text instead.
# The released dataset contains 6 identical texts in both splits; this test
# pins that documented number so any silent data change fails loudly.
KNOWN_TRAIN_TEST_OVERLAP = 6


def test_train_test_text_overlap_is_the_documented_six(dataset):
    normalize = lambda texts: {t.strip().lower() for t in texts}
    overlap = normalize(dataset["train"]["text"]) & \
        normalize(dataset["test"]["text"])
    assert len(overlap) == KNOWN_TRAIN_TEST_OVERLAP
