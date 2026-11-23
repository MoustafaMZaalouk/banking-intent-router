"""API contract tests: valid requests, invalid input, abstention."""

import pytest
from fastapi.testclient import TestClient

from src.api import create_app


@pytest.fixture()
def client(tiny_model_dir):
    return TestClient(create_app(model_dir=tiny_model_dir))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_intents_endpoint(client, tiny_model_dir):
    body = client.get("/intents").json()
    assert body["count"] == 3
    assert "card_lost" in body["intents"]


def test_predict_valid_request(client):
    response = client.post("/predict", json={"text": "i lost my card"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "card_lost"
    assert 0.0 < body["confidence"] <= 1.0
    assert body["abstained"] is False


def test_predict_with_explicit_model_and_threshold(client):
    response = client.post("/predict", json={
        "text": "rate for euros", "model": "tfidf",
        "abstain_threshold": 1.0})
    assert response.status_code == 200
    assert response.json()["abstained"] is True


@pytest.mark.parametrize("payload", [
    {},                                   # missing text
    {"text": ""},                         # empty text
    {"text": "x" * 1001},                 # too long
    {"text": "hi", "model": "nonsense"},  # unknown model
    {"text": "hi", "abstain_threshold": 1.5},   # threshold out of range
    {"text": "hi", "abstain_threshold": -0.1},
    {"text": 42},                         # wrong type
])
def test_predict_invalid_requests_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_embeddings_model_unavailable_in_tiny_deployment(client):
    response = client.post("/predict",
                           json={"text": "hello", "model": "embeddings"})
    assert response.status_code == 503
