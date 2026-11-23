"""Router inference correctness on a tiny trained model."""

import pytest

from src.predict import IntentRouter


def test_predict_returns_learned_intent(tiny_model_dir):
    router = IntentRouter(tiny_model_dir)
    result = router.predict("i think i lost my card", model="tfidf")
    assert result["intent"] == "card_lost"
    assert 0.0 < result["confidence"] <= 1.0
    assert result["abstained"] is False
    assert result["model"] == "tfidf"


def test_predict_abstains_when_threshold_is_one(tiny_model_dir):
    router = IntentRouter(tiny_model_dir)
    result = router.predict("exchange rate for yen", model="tfidf",
                            abstain_threshold=1.0)
    assert result["abstained"] is True
    assert result["intent"] == "exchange_rate"  # best guess still reported


def test_predict_rejects_bad_input(tiny_model_dir):
    router = IntentRouter(tiny_model_dir)
    with pytest.raises(ValueError):
        router.predict("", model="tfidf")
    with pytest.raises(ValueError):
        router.predict("   ", model="tfidf")
    with pytest.raises(ValueError):
        router.predict("hello", model="nonsense")
    with pytest.raises(ValueError):
        router.predict("hello", model="embeddings")  # not in tiny fixture
