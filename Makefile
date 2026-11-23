PY ?= python3

.PHONY: install data train eval plots benchmark test lint smoke all

# --extra-index-url serves the smaller CPU-only torch wheel (2.14.1+cpu);
# without it pip installs the larger PyPI CUDA build of the same version.
install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt \
		--extra-index-url https://download.pytorch.org/whl/cpu

data:
	$(PY) scripts/download_data.py

train:
	$(PY) scripts/train_models.py

eval:
	$(PY) scripts/evaluate.py
	$(PY) scripts/make_plots.py
	$(PY) scripts/benchmark.py

export-onnx:
	$(PY) scripts/export_onnx.py

benchmark-onnx:
	$(PY) scripts/benchmark_onnx.py

test:
	$(PY) -m pytest tests/ -m "not slow"

test-all:
	$(PY) -m pytest tests/

lint:
	$(PY) -m ruff check src/ scripts/ tests/

smoke:
	$(PY) scripts/smoke_train.py

all: data train eval
