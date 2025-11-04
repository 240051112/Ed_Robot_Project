# ed_core/rag_service/emb_onnx.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import List, Iterable

class OnnxSentenceEmbedder:
    """
    Default: sentence-transformers/all-MiniLM-L6-v2 in ONNX form
    Directory layout expected:
      /path/to/model/
        - model.onnx            (BERT/MiniLM encoder exported to ONNX)
        - tokenizer.json        (HF fast tokenizer)
        - config.json           (optional)
    """
    def __init__(self, model_dir: Path, use_gpu: bool = False):
        self.model_dir = Path(model_dir)
        tok_path = self.model_dir / "tokenizer.json"
        onnx_path = self.model_dir / "model.onnx"
        assert tok_path.exists(), f"tokenizer.json not found in {self.model_dir}"
        assert onnx_path.exists(), f"model.onnx not found in {self.model_dir}"

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        # typical max len for MiniLM (works fine for chunk text)
        self.max_len = 256

    def _encode(self, texts: List[str]) -> np.ndarray:
        # fast tokenizer outputs to NumPy
        input_ids = []
        attention = []
        for t in texts:
            enc = self.tokenizer.encode(t)
            ids = enc.ids[: self.max_len]
            att = [1] * len(ids)
            # pad
            pad = self.max_len - len(ids)
            if pad > 0:
                ids = ids + [0] * pad
                att = att + [0] * pad
            input_ids.append(ids)
            attention.append(att)

        inputs = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention, dtype=np.int64),
        }
        out = self.session.run(None, inputs)[0]  # shape [B, T, H] last_hidden_state
        # mean pool with attention mask
        mask = inputs["attention_mask"][:, :, None].astype(np.float32)  # [B,T,1]
        summed = (out * mask).sum(axis=1)                               # [B,H]
        counts = np.clip(mask.sum(axis=1), 1e-9, None)                  # [B,1]
        emb = summed / counts
        # L2 normalize
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norm
        return emb.astype(np.float32)

    def embed(self, items: Iterable[str]) -> List[List[float]]:
        X = self._encode(list(items))
        return X.tolist()
