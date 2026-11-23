"""Feature correctness: TF-IDF math and embedding determinism."""

import numpy as np
import pytest

from src.features import EmbeddingExtractor, TfidfExtractor


def test_tfidf_idf_math():
    """A term in every document gets idf 1 (log(1)+1 with smoothing)."""
    extractor = TfidfExtractor(max_features=100, ngram_range=(1, 1))
    extractor.fit_transform(["shared one", "shared two"])
    idf = extractor.vectorizer.idf_
    vocab = extractor.vectorizer.vocabulary_
    # idf("shared") = log((1+2)/(1+2)) + 1 = 1 (smooth_idf default)
    assert idf[vocab["shared"]] == pytest.approx(1.0)
    # "one" appears in 1 of 2 docs: idf = log(3/2) + 1
    assert idf[vocab["one"]] == pytest.approx(np.log(3 / 2) + 1)


def test_tfidf_uses_only_train_vocabulary():
    extractor = TfidfExtractor(max_features=100)
    extractor.fit_transform(["train words only"])
    transformed = extractor.transform(["unseen token"])
    assert transformed.nnz == 0  # no leakage, no crash on unseen words


@pytest.mark.slow
def test_embedding_dim_and_determinism():
    embedder = EmbeddingExtractor()
    first = embedder.transform(["where is my card"])
    second = embedder.transform(["where is my card"])
    other = embedder.transform(["how do i top up my account"])
    assert first.shape == (1, 384)
    np.testing.assert_allclose(first, second, atol=1e-5)
    cosine = float(first @ other.T)
    assert cosine < 0.9  # different intents embed differently
