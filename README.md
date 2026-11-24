# Banking Intent Router

[![CI](https://github.com/MoustafaMZaalouk/banking-intent-router/workflows/CI/badge.svg)](https://github.com/MoustafaMZaalouk/banking-intent-router/blob/main/.github/workflows/ci.yml)

*Live workflow status, not a production-readiness or model-quality rating.*

Classifies customer-service queries into the 77 intents of the real
[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) dataset.
Compares a TF-IDF baseline against frozen pretrained sentence embeddings,
ships a FastAPI service with confidence-based abstention, and measures
everything it claims.

## Honesty notes (read first)

- This project was built with AI assistance (Instinct agent), and this
  README says so because pretending otherwise would be dishonest.
- The sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
  is used **frozen**. Nothing about it was trained or fine-tuned here.
  The ONNX variant is the same frozen model, int8-quantized.
- The only things actually trained in this repo are two scikit-learn
  logistic-regression classifiers, one per feature type.
- Fine-tuning a transformer (e.g. SetFit) was **not** done: full-data
  transformer fine-tuning on CPU is not realistically reproducible in a
  student project, and faking it would be dishonest. Published
  fine-tuned transformer results on BANKING77 are around 92-93%
  accuracy; everything below is what this repo actually measures.
- All reported numbers come from committed scripts (see
  "Reproduce"). Nothing is hand-typed.
- Dataset hygiene: BANKING77 contains 6 identical texts in both the
  train and test splits. This is pinned by
  `tests/test_data.py::test_train_test_text_overlap_is_the_documented_six`
  so any silent data change fails loudly.

## Results (BANKING77 test split, 3,080 queries)

From `results/metrics.json`, produced by `scripts/evaluate.py`:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| TF-IDF + logistic regression | 0.857 | 0.856 |
| Frozen MiniLM embeddings + logistic regression | 0.908 | 0.908 |

Frozen features plus logistic regression close most of the gap to the
published ~92-93% fine-tuned transformer range, at a fraction of the
cost.

Inference latency (single request, CPU), from `results/latency.json` and
`results/latency_onnx.json`, produced by `scripts/benchmark.py` and
`scripts/benchmark_onnx.py`:

| Serving path | Mean | p95 | Throughput |
| --- | --- | --- | --- |
| TF-IDF + LR | 2.13 ms | 2.18 ms | 468.9 req/s |
| Frozen MiniLM (PyTorch) + LR | 10.8 ms | 12.59 ms | 92.6 req/s |
| Frozen MiniLM (int8 ONNX) + LR | 2.05 ms | 2.9 ms | 486.2 req/s |

The quantized ONNX embedder keeps embeddings faithful (cosine similarity
0.96822 vs PyTorch) and reaches
0.9023 test accuracy with the same
classifier, roughly 5x faster than the PyTorch path on this machine.

Training time (`results/runtime_train.json`, x86_64 CPU): TF-IDF
features 0.13 s + fit 3.0 s; embedding encoding 49.9 s (train) + 14.8 s
(test) + fit 0.77 s. The embedding model download is a one-time cost.

### Confidence-based abstention

The router answers only when its top-class probability clears a
threshold; otherwise it abstains and a human should take over. From
`results/abstention.csv` (embeddings model): at 91% coverage accuracy
rises to 0.947; at 80% coverage, 0.971. The trade-off curve is in
`plots/abstention_curve.png`.

### Error analysis

`results/error_analysis.md` lists the top confused intent pairs with
real example queries. The hardest pairs are genuinely close intents:
`virtual_card_not_working` vs `get_disposable_virtual_card`,
`why_verify_identity` vs `verify_my_identity`, and
`request_refund` vs `Refund_not_showing_up`. Confusion matrices for both
models are in `plots/confusion_*.png` and `results/confusion_*.csv`.

## How it works

1. **Features.** Each query is turned into numbers two ways. TF-IDF
   counts weighted word unigrams/bigrams (vocabulary learned from the
   training split only). The embedding path runs the query through the
   frozen MiniLM sentence transformer, which maps any sentence to a
   384-dimension vector where similar meanings land close together.
2. **Classifier.** A logistic-regression classifier learns a weight per
   intent and outputs a probability for each of the 77 intents. This is
   the only trained component.
3. **Abstention.** If the top probability is below the threshold, the
   API says `abstained: true` instead of guessing.
4. **API.** `src/api.py` serves `/predict`, `/intents`, and `/health`.
   Model choice per request: `tfidf`, `embeddings`, or `embeddings-onnx`.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # pinned deps; pulls the CPU torch wheel via
                    # --extra-index-url https://download.pytorch.org/whl/cpu
make data           # download BANKING77 into data/banking77
make train          # train both classifiers -> models/ (checkpointed)
make eval           # metrics, plots, latency -> results/ and plots/
make test           # fast test suite
make lint           # ruff
```

Verified end to end from a fresh virtualenv (Python 3.10) with exactly the
commands above.

Optional ONNX embedder (needs a separate venv, see
`scripts/export_onnx.py` docstring):

```bash
make export-onnx
pip install -r requirements-onnx.txt
make benchmark-onnx
```

Run the API after training:

```bash
uvicorn src.api:app --reload
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I lost my card", "model": "embeddings"}'
```

## Project layout

```
src/            data loading, features, training, evaluation, router, API
scripts/        download/train/evaluate/plot/benchmark/ONNX-export scripts
tests/          pytest suite (dataset validation, metric math, API contract)
results/        metrics.json, confusion matrices, abstention curve, latency
plots/          regenerable figures (make eval)
models/         trained artifacts (gitignored, rebuilt by make train)
```

## Future work

- SetFit few-shot comparison: train SetFit on a few examples per intent
  and compare against the frozen-feature classifiers.

## Credits and licenses

- **Code in this repo:** MIT (see LICENSE). Copyright Moustafa Zaalouk.
- **Dataset:** [BANKING77](https://huggingface.co/datasets/PolyAI/banking77),
  CC BY 4.0. Citation: Casanueva et al., "Efficient Intent Detection with
  Dual Sentence Encoders", Proceedings of the 2nd Workshop on NLP for
  ConvAI - ACL 2020, https://arxiv.org/abs/2003.04807
- **Embedding model:**
  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  Apache 2.0. Used frozen; ONNX variant is the same weights int8-quantized.
