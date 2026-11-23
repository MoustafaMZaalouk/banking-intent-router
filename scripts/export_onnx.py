"""Export the frozen MiniLM embedder to a quantized ONNX model.

The ONNX toolchain (optimum) currently requires transformers<4.58 while
the main environment uses transformers 5.x, so run this in a separate venv:

    python3 -m venv /tmp/onnxenv
    /tmp/onnxenv/bin/pip install "optimum[onnxruntime]"
    /tmp/onnxenv/bin/python scripts/export_onnx.py

The result is models/embedder_onnx/model_quantized.onnx (~23 MB vs ~90 MB
PyTorch). The embedding weights are still frozen - quantization compresses
them, it does not train them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

EXPORT_DIR = Path("/tmp/onnx_export")
OUT_DIR = Path("models/embedder_onnx")


def main():
    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", export=True)
    model.save_pretrained(EXPORT_DIR)
    quantizer = ORTQuantizer.from_pretrained(EXPORT_DIR)
    config = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    quantizer.quantize(save_dir=OUT_DIR, quantization_config=config)
    print(f"quantized ONNX embedder -> {OUT_DIR}")


if __name__ == "__main__":
    main()
