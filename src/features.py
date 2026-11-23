"""Feature extractors for the intent router.

Two views of the same text are compared in this project:

1. ``TfidfExtractor`` - sparse bag-of-words TF-IDF features computed from
   the training split only (the classic baseline).
2. ``EmbeddingExtractor`` - dense sentence embeddings from the pretrained
   ``sentence-transformers/all-MiniLM-L6-v2`` model. The model weights are
   frozen: we only call ``encode`` and never train or fine-tune it.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TfidfExtractor:
    """TF-IDF features over word unigrams and bigrams."""

    def __init__(self, max_features=20000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)


class EmbeddingExtractor:
    """Frozen sentence-embedding features from a pretrained model."""

    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        # Imported lazily so that using only the TF-IDF path never pays
        # the import cost of torch / sentence-transformers.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def transform(self, texts, batch_size=64):
        """Encode texts into dense vectors; weights stay frozen."""
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
