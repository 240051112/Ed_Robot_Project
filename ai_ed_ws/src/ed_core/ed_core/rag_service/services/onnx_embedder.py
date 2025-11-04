# services/onnx_embedder.py
import numpy as np, onnxruntime as ort, re
from typing import List

class OnnxEmbedder:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.ip = self.sess.get_inputs()[0].name
        self.att = self.sess.get_inputs()[1].name
        self.op = self.sess.get_outputs()[0].name
        # very light tokenizer (WordPiece/BERT-like): for demo, use whitespace
        # replace with a tiny-bert tokenizer if you have it locally
    def encode(self, texts: List[str]) -> np.ndarray:
        # Replace with your local tokenizer. For now, mean-pool fake IDs.
        # (If you’ve got a saved tokenizer, load it and build input_ids/attention_mask properly.)
        arr = np.stack([np.mean([len(w) for w in re.findall(r"\w+", t)] or [1.0]) * np.ones(384) for t in texts])
        return arr.astype(np.float32)

def cosine(a, b):  # a:[n,d], b:[m,d]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T
