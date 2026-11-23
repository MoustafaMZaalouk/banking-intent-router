"""A small FastAPI service that exposes the trained intent router.

Run it (after training) with:

    uvicorn src.api:app --reload

and POST JSON like {"text": "I lost my card", "model": "tfidf"}
to /predict. The service answers only when the model is confident enough;
otherwise it abstains and a human should take over.
"""

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import IntentRouter

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    model: Literal["tfidf", "embeddings", "embeddings-onnx"] = "tfidf"
    abstain_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    text: str
    model: str
    intent: str
    confidence: float
    abstained: bool
    abstain_threshold: float


def create_app(model_dir=None):
    """Build the FastAPI app around a trained model directory."""
    if model_dir is None:
        model_dir = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))

    app = FastAPI(title="Banking Intent Router", version="1.0.0")
    state = {"router": None}

    def get_router():
        if state["router"] is None:
            try:
                state["router"] = IntentRouter(model_dir)
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="model artifacts not found; run scripts/train_models.py first",
                ) from exc
        return state["router"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/intents")
    def intents():
        router = get_router()
        return {"count": len(router.label_names), "intents": router.label_names}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        router = get_router()
        if request.model == "embeddings" \
                and router.embedding_classifier is None:
            raise HTTPException(
                status_code=503,
                detail="embedding classifier is not available in this deployment",
            )
        if request.model == "embeddings-onnx" and not router.has_onnx:
            raise HTTPException(
                status_code=503,
                detail="ONNX embedder is not available in this deployment",
            )
        try:
            return router.predict(
                request.text,
                model=request.model,
                abstain_threshold=request.abstain_threshold,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
