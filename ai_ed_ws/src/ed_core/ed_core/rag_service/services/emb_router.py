# ~/ai_ed_ws/src/ed_core/ed_core/rag_service/services/emb_router.py
from fastapi import APIRouter
from pydantic import BaseModel
import os, sys, traceback

# ensure we pick up user-site where safetensors.torch lives on Jetson
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.10/site-packages"))

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMB_MODEL", "thenlper/gte-small")  # 384-dim, public
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

router = APIRouter()
_model = None  # lazy init


def get_model():
    global _model
    if _model is None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HUB_ENABLE_IPv6", "0")
        _model = SentenceTransformer(MODEL_ID, device=DEVICE)
    return _model


class Req(BaseModel):
    content: str


@router.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "model": MODEL_ID}


@router.post("/embedding")
def embedding(req: Req):
    try:
        m = get_model()
        vec = m.encode(
            req.content,
            normalize_embeddings=True,   # cosine-sim ready
            convert_to_numpy=True
        )
        return {
            "embedding": vec.tolist(),
            "dim": int(len(vec)),
            "device": DEVICE,
            "model": MODEL_ID
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
