# ed_core/rag_service/config.py
from __future__ import annotations
from pathlib import Path
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ament_index_python.packages import get_package_share_directory
import importlib.resources as pkgres

def _expand(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    return Path(os.path.expanduser(str(p))).resolve()

def _default_persona() -> Path:
    try:
        return Path(pkgres.files("ed_core.rag_service") / "system_prompt.md")
    except Exception:
        return Path(__file__).with_name("system_prompt.md")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ED_", case_sensitive=False, extra="ignore")

    # -------- Paths --------
    docs_dir: Path | None = Field(default=None)
    chroma_db_dir: Path = Field(default=Path.home() / ".local/share/ed/chroma_db")
    model_path: Path = Field(default=Path("/home/jetson/phi3_models/Phi-3-mini-4k-instruct-q4.gguf"))
    embedding_local_dir: Path = Field(default=Path("/home/jetson/models/hf/all-MiniLM-L6-v2"))
    reranker_local_dir: Path = Field(default=Path("/home/jetson/models/hf/ms-marco-MiniLM-L-6-v2"))
    persona_path: Path = Field(default_factory=_default_persona)

    # -------- Behavior toggles --------
    allow_open: bool = Field(default=False)
    rebuild_on_start: bool = Field(default=False)

    # -------- RAG / Answering knobs --------
    query_max_tokens: int = Field(default=110)
    query_temp: float = Field(default=0.2)
    rag_top_k: int = Field(default=6)
    rag_threshold: float = Field(default=0.30)
    rag_min_docs: int = Field(default=1)
    rerank_top_m: int = Field(default=3)

    # NEW: embedding backend ("tfidf" | "onnx")
    embedding_backend: str = Field(default="tfidf")
    onnx_model_dir: Path = Field(default=Path("/home/jetson/models/hf/all-MiniLM-L6-v2-onnx"))
    onnx_use_gpu: int = Field(default=0)  # 1 to try CUDA EP

    # Also surface these so /admin/info isn't null when you curl it
    use_llm_summary: int = Field(default=1)
    strict_rag: int = Field(default=0)
    rulebook_fallback: int = Field(default=0)
    timeout_s: int = Field(default=25)

    # -------- LLM runtime knobs --------
    ctx_size: int = Field(default=4096)
    n_batch: int = Field(default=128)
    n_gpu_layers: int = Field(default=-1)
    seed: int = Field(default=0)
    use_mlock: int = Field(default=0)
    llm_retries: int = Field(default=2)

    def __init__(self, **data):
        super().__init__(**data)
        self.chroma_db_dir = _expand(self.chroma_db_dir)
        self.model_path = _expand(self.model_path)
        self.embedding_local_dir = _expand(self.embedding_local_dir)
        self.reranker_local_dir = _expand(self.reranker_local_dir)
        self.onnx_model_dir = _expand(self.onnx_model_dir)
        if self.docs_dir is not None:
            self.docs_dir = _expand(self.docs_dir)

    @property
    def documents_dir(self) -> Path:
        if self.docs_dir:
            return self.docs_dir
        share = Path(get_package_share_directory("ed_core"))
        return share / "documents"

settings = Settings()
