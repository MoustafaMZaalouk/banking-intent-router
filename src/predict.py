"""Inference: load trained artifacts and route new queries."""

import json
from pathlib import Path

import joblib

from src.evaluate import decide
from src.features import EmbeddingExtractor

DEFAULT_THRESHOLD = 0.5
MODEL_CHOICES = ("tfidf", "embeddings", "embeddings-onnx")


class IntentRouter:
    """Routes a customer query to one of the trained banking intents."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "config.json", encoding="utf-8") as handle:
            self.config = json.load(handle)
        with open(model_dir / "label_names.json", encoding="utf-8") as handle:
            self.label_names = json.load(handle)

        self.tfidf_extractor = joblib.load(model_dir / "tfidf_extractor.joblib")
        self.tfidf_classifier = joblib.load(model_dir / "tfidf_classifier.joblib")

        embedding_path = model_dir / "embedding_classifier.joblib"
        self.embedding_classifier = (
            joblib.load(embedding_path) if embedding_path.exists() else None
        )
        # The sentence-embedding model is downloaded/loaded lazily, so the
        # TF-IDF path works even without torch installed.
        self._embedder = None
        self._onnx_encoder = None
        self.onnx_path = model_dir / "embedder_onnx" / "model_quantized.onnx"
        self.has_onnx = self.onnx_path.exists()

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingExtractor(self.config["embedding_model"])
        return self._embedder

    def predict_proba(self, text, model="tfidf"):
        """Class-probability vector for one query from the chosen model."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if model == "tfidf":
            features = self.tfidf_extractor.transform([text])
            return self.tfidf_classifier.predict_proba(features)[0]
        if model == "embeddings":
            if self.embedding_classifier is None:
                raise ValueError("embedding classifier is not available")
            features = self._get_embedder().transform([text])
            return self.embedding_classifier.predict_proba(features)[0]
        if model == "embeddings-onnx":
            if self.embedding_classifier is None or not self.has_onnx:
                raise ValueError("ONNX embedder is not available")
            features = self._get_onnx_encoder()([text])
            return self.embedding_classifier.predict_proba(features)[0]
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_CHOICES}")

    def _get_onnx_encoder(self):
        if self._onnx_encoder is None:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer

            session = ort.InferenceSession(str(self.onnx_path))
            tokenizer = AutoTokenizer.from_pretrained(
                self.config["embedding_model"])

            def encode(texts):
                encoded = tokenizer(list(texts), padding=True,
                                    truncation=True, max_length=512,
                                    return_tensors="np")
                embeddings = session.run(None, dict(encoded))[0]
                if embeddings.ndim == 3:  # last_hidden_state -> mean pool
                    mask = encoded["attention_mask"][..., None]
                    embeddings = (embeddings * mask).sum(axis=1) / \
                        np.maximum(mask.sum(axis=1), 1)
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                return embeddings / np.maximum(norm, 1e-12)

            self._onnx_encoder = encode
        return self._onnx_encoder

    def predict(self, text, model="tfidf", abstain_threshold=None):
        """Route one query; abstain when confidence is too low."""
        if abstain_threshold is None:
            abstain_threshold = DEFAULT_THRESHOLD
        probabilities = self.predict_proba(text, model=model)
        best_index, confidence, abstained = decide(probabilities,
                                                   abstain_threshold)
        return {
            "text": text,
            "model": model,
            "intent": self.label_names[best_index],
            "confidence": round(confidence, 4),
            "abstained": abstained,
            "abstain_threshold": abstain_threshold,
        }
