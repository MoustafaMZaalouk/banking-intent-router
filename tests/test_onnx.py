"""ONNX embedder integration tests.

Marked slow: they need models/embedder_onnx/ (run scripts/export_onnx.py)
and download the tokenizer. Skipped when the artifact is absent.
"""

from pathlib import Path

import numpy as np
import pytest

from src.predict import IntentRouter

pytestmark = pytest.mark.slow

REPO_MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture()
def router():
    if not (REPO_MODELS / "embedder_onnx" / "model_quantized.onnx").exists():
        pytest.skip("quantized ONNX embedder not exported")
    return IntentRouter(REPO_MODELS)


def test_onnx_matches_pytorch_embedding(router):
    text = "where is my new card?"
    pytorch_features = router._get_embedder().transform([text])
    onnx_features = router._get_onnx_encoder()([text])
    cosine = float(np.sum(pytorch_features[0] * onnx_features[0]))
    assert cosine > 0.95  # int8 quantization keeps embeddings faithful


def test_onnx_predict_routes_card_query(router):
    result = router.predict("my card never arrived", model="embeddings-onnx")
    reference = router.predict("my card never arrived", model="embeddings")
    assert result["intent"] == "card_arrival"
    assert result["intent"] == reference["intent"]  # matches PyTorch path
