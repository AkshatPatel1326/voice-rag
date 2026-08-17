"""
Lightweight query embedder using ONNX Runtime instead of full PyTorch +
sentence-transformers. Used only for deployment (Render free tier, 512MB RAM).

Produces embeddings numerically very close to the original sentence-transformers
model, since it's exported and quantized from the exact same checkpoint used to
build the FAISS index in Phase 2 (small quantization noise, negligible effect
on retrieval quality).

The quantized model + tokenizer are hosted on the Hugging Face Hub (not in this
git repo — the quantized file is ~112MB, just over GitHub's 100MB file limit)
and downloaded once at container startup, then cached locally.
"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# Change this to your actual HF username/repo from step 1-2 above.
HF_REPO_ID = "Akshat1326/voice-rag-onnx-embedder"
MAX_LENGTH = 128


class OnnxQueryEmbedder:
    """Drop-in replacement for SentenceTransformer's .encode() for this one use
    case (embedding a single short query string), so harness.py's _retrieve()
    doesn't need to change."""

    def __init__(self, repo_id: str = HF_REPO_ID):
        model_path = hf_hub_download(repo_id=repo_id, filename="model_quantized.onnx")
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")

        # Keep onnxruntime's own memory footprint as small as possible: by
        # default it pre-allocates a generous memory arena and spins up
        # multiple threads, sized for a server-class machine. On Render's
        # 512MB free tier that overhead alone can tip us over the limit, so
        # we disable the arena and cap threads to 1 (the model is tiny and
        # we're only ever embedding one short query at a time anyway).
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self._input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, texts, normalize_embeddings: bool = True):
        encodings = [self.tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)
        onnx_inputs = {k: v for k, v in onnx_inputs.items() if k in self._input_names}

        outputs = self.session.run(None, onnx_inputs)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden)

        # Mean pooling over tokens, weighted by attention mask (same as
        # sentence-transformers' default pooling for this model).
        mask = attention_mask[..., None].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = summed / counts

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)

        return embeddings.astype("float32")