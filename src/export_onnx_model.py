"""
One-time LOCAL export: converts the sentence-transformers embedding model to a
quantized ONNX model, so the deployed backend (Render free tier, 512MB RAM) can
embed queries with lightweight `onnxruntime` instead of full PyTorch +
sentence-transformers, which is too heavy for the free tier (float32 model
weights alone are ~470MB; PyTorch's own runtime overhead adds another
~250-300MB on top of that).

Run this ONCE on your local machine (needs extra packages, fine here since you
have plenty of RAM â€” these packages do NOT need to be installed on the server):

    pip install optimum[onnxruntime] --break-system-packages
    python src/export_onnx_model.py

This produces data/onnx_model/ (a quantized ~120MB model + tokenizer files).
Commit that folder to git â€” the deployed server loads it directly, no export
step happens on the server itself.

Only affects QUERY embedding at inference time. The FAISS index itself (built
in Phase 2 from the full-precision model) is unchanged â€” no need to rebuild it.
Quantization introduces a small amount of numerical noise, which in practice
has a negligible effect on nearest-neighbor retrieval quality.
"""

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAVE_DIR = "data/onnx_model"

print(f"Exporting {MODEL_ID} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Unquantized ONNX model saved to {SAVE_DIR}/")

print("Quantizing to int8 (this shrinks the model ~4x)...")
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
quantizer.quantize(save_dir=SAVE_DIR, quantization_config=qconfig)

print(f"Done. Quantized model saved to {SAVE_DIR}/model_quantized.onnx")
print("Commit the whole data/onnx_model/ folder to git.")