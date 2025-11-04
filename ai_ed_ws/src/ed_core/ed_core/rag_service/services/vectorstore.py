from __future__ import annotations
import os, shutil, re, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- Optional PDF extractor (PyMuPDF) ----
_HAVE_FITZ = False
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False

from ..config import settings

# ==============================
# Backends & on-disk structures
# ==============================
INDEX_DIRNAME_TFIDF = "tfidf_index"
VEC_FILE      = "vectorizer.joblib"
MATRIX_FILE   = "matrix.joblib"
METAS_FILE    = "metas.joblib"
TEXTS_FILE    = "texts.joblib"

SEMANTIC_DIRNAME = "chroma_semantic"
CHROMA_COLLECTION = "ed_semantic"

# Optional TF-IDF tuning via env
_MIN_DF = int(os.getenv("ED_TFIDF_MIN_DF", "1"))
_MAX_DF = os.getenv("ED_TFIDF_MAX_DF", "1.0")
try:
    _MAX_DF = float(_MAX_DF)
except Exception:
    _MAX_DF = 1.0

# =========================
# Utilities used by API
# =========================
def chroma_dir_global(db_dir: Path | None = None) -> Path:
    return Path(db_dir) if db_dir is not None else settings.chroma_db_dir

def ensure_dir_writable(path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    testfile = path / ".write_test"
    testfile.write_text("ok", encoding="utf-8")
    try:
        testfile.unlink(missing_ok=True)
    except Exception:
        pass

def drop_db(db_dir: Path) -> None:
    if db_dir.exists():
        shutil.rmtree(db_dir, ignore_errors=True)

def db_size_bytes(db_dir: Path) -> int:
    total = 0
    for p in Path(db_dir).rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def gpu_info() -> Dict[str, Any]:
    # Retriever itself is CPU unless ONNX CUDA EP is enabled
    return {"cuda": bool(getattr(settings, "onnx_use_gpu", 0)), "name": "onnx" if getattr(settings, "embedding_backend", "tfidf") == "onnx" else "cpu"}

# ---------------------------
# Cleaning & chunking helpers
# ---------------------------
_HEADER_FOOTER_PATTERNS = [
    r"©\s*\d{4}[^.\n]+(All rights reserved|Rights reserved).*",
    r"All\s+rights\s+reserved.*",
    r"Page\s*\d+\s*(of\s*\d+)?",
    r"\b(confidential|copyright|proprietary)\b.*",
    r"https?://\S+",
    r"\bTable\s+of\s+Contents\b.*",
]

def _clean_text_block(text: str) -> str:
    if not text:
        return ""
    t = text
    for pat in _HEADER_FOOTER_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if len(re.sub(r"[A-Za-z0-9]", "", s)) > len(re.sub(r"[^A-Za-z0-9]", "", s))*2:
            continue
        if len(s) < 4:
            continue
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _split_into_paragraphs(clean_page: str) -> List[str]:
    if not clean_page:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", clean_page) if p.strip()]
    if len(paras) >= 1:
        return paras
    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", clean_page)
    out, buf, count = [], [], 0
    for s in parts:
        buf.append(s)
        count += 1
        if count >= 3:
            out.append(" ".join(buf).strip())
            buf, count = [], 0
    if buf:
        out.append(" ".join(buf).strip())
    return out

def _good_chunk(text: str) -> bool:
    s = text.strip()
    if len(s) < 80:
        return False
    if len(s) > 2400:
        return False
    letters = len(re.findall(r"[A-Za-z]", s))
    dots = s.count(".")
    if letters > 0 and dots > letters*1.2:
        return False
    upp = sum(1 for c in s if c.isupper())
    if letters > 40 and upp > letters*0.85:
        return False
    return True

def _extract_pdf_texts(pdf_path: Path) -> List[Tuple[str, int, str]]:
    """
    Return list[(source, page_number, chunk_text)]
    """
    out: List[Tuple[str, int, str]] = []
    if not _HAVE_FITZ:
        return out
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            raw_text = page.get_text("text")
            cleaned = _clean_text_block(raw_text)
            if not cleaned:
                continue
            paras = _split_into_paragraphs(cleaned)
            for chunk in paras:
                if _good_chunk(chunk):
                    out.append((pdf_path.name, i + 1, chunk))
    except Exception:
        pass
    return out

# ============================
# TF-IDF implementation (old)
# ============================
class _Index:
    def __init__(self, idx_dir: Path):
        self.idx_dir = idx_dir
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None  # scipy CSR
        self.metas: List[Dict[str, Any]] = []
        self.texts: List[str] = []

# ============================
# ONNX embedder (semantic)
# ============================
class _OnnxSentenceEmbedder:
    """
    Lightweight ONNX encoder + HF tokenizer.json (no HF framework).
    Expects directory with:
      - model.onnx
      - tokenizer.json
    """
    def __init__(self, model_dir: Path, use_gpu: bool = False):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        tok_path = self.model_dir / "tokenizer.json"
        onnx_path = self.model_dir / "model.onnx"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {self.model_dir}")
        if not onnx_path.exists():
            raise FileNotFoundError(f"model.onnx not found in {self.model_dir}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.max_len = 256

    def _encode(self, texts: List[str]) -> np.ndarray:
        input_ids, attention = [], []
        for t in texts:
            enc = self.tokenizer.encode(t)
            ids = enc.ids[: self.max_len]
            att = [1] * len(ids)
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
        out = self.session.run(None, inputs)[0]  # [B,T,H] last_hidden_state
        mask = inputs["attention_mask"][:, :, None].astype(np.float32)
        summed = (out * mask).sum(axis=1)               # [B,H]
        counts = np.clip(mask.sum(axis=1), 1e-9, None)  # [B,1]
        emb = summed / counts
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return emb.astype(np.float32)

    def embed(self, items: Iterable[str]) -> List[List[float]]:
        X = self._encode(list(items))
        return X.tolist()

# ============================
# VectorStore (switchable)
# ============================
class VectorStore:
    def __init__(self):
        self.docs_dir = Path(settings.documents_dir)

        self.backend = (getattr(settings, "embedding_backend", "tfidf") or "tfidf").lower()
        self.idx_dir_tfidf = Path(chroma_dir_global()) / INDEX_DIRNAME_TFIDF
        self.index = _Index(self.idx_dir_tfidf)
        self._page_map: Dict[Tuple[str, int], str] = {}

        # Semantic bits (lazy)
        self.semantic_ok = False
        self._chroma_client = None
        self._chroma_collection = None
        self._onnx = None
        self.idx_dir_sem = Path(chroma_dir_global()) / SEMANTIC_DIRNAME

        if self.backend == "onnx":
            try:
                from chromadb import PersistentClient
                from chromadb.utils import embedding_functions  # noqa: F401  (kept for type compat)
                self._onnx = _OnnxSentenceEmbedder(
                    model_dir=Path(getattr(settings, "onnx_model_dir")),
                    use_gpu=bool(getattr(settings, "onnx_use_gpu", 0)),
                )
                self.idx_dir_sem.mkdir(parents=True, exist_ok=True)
                self._chroma_client = PersistentClient(path=str(self.idx_dir_sem))

                class _EF:
                    def __init__(self, onnx: _OnnxSentenceEmbedder):
                        self._onnx = onnx
                    def __call__(self, input: List[str]) -> List[List[float]]:
                        return self._onnx.embed(input)

                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_EF(self._onnx),
                )
                self.semantic_ok = True
            except Exception:
                # Fall back to TF-IDF if anything goes wrong
                self.backend = "tfidf"
                self.semantic_ok = False

    # ---- Common status for /admin ----
    def status(self) -> Dict[str, Any]:
        if self.backend == "onnx":
            size = db_size_bytes(self.idx_dir_sem)
            return {
                "backend": "onnx",
                "index_dir": str(self.idx_dir_sem),
                "size_bytes": size,
                "docs_dir": str(self.docs_dir),
                "have_fitz": _HAVE_FITZ,
                "semantic_ok": bool(self.semantic_ok),
            }
        else:
            exists = (self.idx_dir_tfidf / VEC_FILE).exists() and (self.idx_dir_tfidf / MATRIX_FILE).exists()
            count = len(self.index.texts) if self.index.texts else 0
            return {
                "backend": "tfidf",
                "exists": exists,
                "count": count,
                "index_dir": str(self.idx_dir_tfidf),
                "size_bytes": db_size_bytes(self.idx_dir_tfidf),
                "docs_dir": str(self.docs_dir),
                "have_fitz": _HAVE_FITZ,
            }

    # ---- Build / load helpers (shared chunker) ----
    def _scan_documents(self) -> List[Tuple[str, int, str]]:
        chunks: List[Tuple[str, int, str]] = []
        if not _HAVE_FITZ:
            return chunks
        for p in sorted(self.docs_dir.rglob("*.pdf")):
            chunks.extend(_extract_pdf_texts(p))
        return chunks

    def _rebuild_page_map(self) -> None:
        self._page_map.clear()
        # TF-IDF path
        for meta, txt in zip(self.index.metas or [], self.index.texts or []):
            key = (meta["source"], int(meta["page"]))
            self._page_map[key] = " ".join((txt or "").split())
        # Semantic path: nothing to pre-fill (we don’t persist per-page windows here)

    # ---- TF-IDF build/load ----
    def _load_tfidf(self) -> bool:
        try:
            self.index.vectorizer = joblib.load(self.idx_dir_tfidf / VEC_FILE)
            self.index.matrix     = joblib.load(self.idx_dir_tfidf / MATRIX_FILE)
            self.index.metas      = joblib.load(self.idx_dir_tfidf / METAS_FILE)
            self.index.texts      = joblib.load(self.idx_dir_tfidf / TEXTS_FILE)
            self._rebuild_page_map()
            return True
        except Exception:
            return False

    def _build_tfidf(self) -> None:
        chunks = self._scan_documents()
        if not chunks:
            self.index.vectorizer = TfidfVectorizer(
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                token_pattern=r"(?u)\b[\w\-/\.]+\b",
                ngram_range=(1, 2),
                max_features=200_000,
            )
            self.index.metas = []
            self.index.texts = []
            self.index.matrix = None
            self.idx_dir_tfidf.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.index.vectorizer, self.idx_dir_tfidf / VEC_FILE, compress=3)
            joblib.dump(self.index.matrix,     self.idx_dir_tfidf / MATRIX_FILE, compress=3)
            joblib.dump(self.index.metas,      self.idx_dir_tfidf / METAS_FILE, compress=3)
            joblib.dump(self.index.texts,      self.idx_dir_tfidf / TEXTS_FILE, compress=3)
            self._rebuild_page_map()
            return

        texts  = [c[2] for c in chunks]
        metas  = [{"source": c[0], "page": c[1]} for c in chunks]

        vec = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[\w\-/\.]+\b",
            ngram_range=(1, 2),
            max_features=200_000,
            min_df=_MIN_DF,
            max_df=_MAX_DF,
        )
        X64 = vec.fit_transform(texts)
        X = X64.astype(np.float32)

        self.index.vectorizer = vec
        self.index.matrix     = X
        self.index.metas      = metas
        self.index.texts      = texts

        self.idx_dir_tfidf.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.index.vectorizer, self.idx_dir_tfidf / VEC_FILE, compress=3)
        joblib.dump(self.index.matrix,     self.idx_dir_tfidf / MATRIX_FILE, compress=3)
        joblib.dump(self.index.metas,      self.idx_dir_tfidf / METAS_FILE, compress=3)
        joblib.dump(self.index.texts,      self.idx_dir_tfidf / TEXTS_FILE, compress=3)
        self._rebuild_page_map()

    # ---- Semantic build ----
    def _build_semantic(self) -> None:
        if not self.semantic_ok:
            return
        # wipe collection
        try:
            self._chroma_client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._chroma_collection._embedding_function,  # reuse
        )

        chunks = self._scan_documents()  # [(src, page, text)]
        if not chunks:
            return

        # Add in batches to keep memory low on Jetson
        B = 256
        ids, docs, metas = [], [], []
        for (src, pg, txt) in chunks:
            ids.append(f"{src}:{pg}:{uuid.uuid4().hex[:8]}")
            docs.append(txt)
            metas.append({"source": src, "page": int(pg)})

            if len(ids) >= B:
                self._chroma_collection.add(ids=ids, documents=docs, metadatas=metas)
                ids, docs, metas = [], [], []
        if ids:
            self._chroma_collection.add(ids=ids, documents=docs, metadatas=metas)

    # ---- Public lifecycle ----
    def ensure_ready(self) -> None:
        if self.backend == "onnx" and self.semantic_ok:
            ensure_dir_writable(self.idx_dir_sem)
            if settings.rebuild_on_start:
                drop_db(self.idx_dir_sem)
                self._build_semantic()
            # nothing else to load
        else:
            ensure_dir_writable(self.idx_dir_tfidf)
            if settings.rebuild_on_start:
                drop_db(self.idx_dir_tfidf)
            if not self._load_tfidf():
                self._build_tfidf()

    def rebuild(self) -> None:
        if self.backend == "onnx" and self.semantic_ok:
            drop_db(self.idx_dir_sem)
            self.idx_dir_sem.mkdir(parents=True, exist_ok=True)
            self._build_semantic()
        else:
            drop_db(self.idx_dir_tfidf)
            self.idx_dir_tfidf.mkdir(parents=True, exist_ok=True)
            self._build_tfidf()

    # ---- Query ----
    def query(self, q: str, k: int = 4, threshold: float = 0.05) -> List[Dict[str, Any]]:
        if self.backend == "onnx" and self.semantic_ok:
            res = self._chroma_collection.query(query_texts=[q], n_results=max(k, 1))
            docs   = res.get("documents", [[]])[0]
            metas  = res.get("metadatas", [[]])[0]
            ids    = res.get("ids", [[]])[0]
            dists  = res.get("distances", [[]])[0]  # cosine distance in newer Chroma
            out: List[Dict[str, Any]] = []
            for i, t, m, d in zip(ids, docs, metas, dists):
                score = 1.0 - float(d)  # convert distance→similarity
                if score < threshold:
                    continue
                out.append({
                    "source": m.get("source", ""),
                    "page": int(m.get("page", 0) or 0),
                    "score": score,
                    "text": t,
                })
                if len(out) >= k:
                    break
            return out

        # TF-IDF fallback
        if self.index.vectorizer is None or self.index.matrix is None:
            return []
        vx = self.index.vectorizer.transform([q]).astype(np.float32)
        sims = linear_kernel(vx, self.index.matrix)[0]
        order = np.argsort(-sims)[: max(k * 4, 1)]
        hits: List[Dict[str, Any]] = []
        for i in order:
            score = float(sims[i])
            if score < threshold:
                continue
            meta = self.index.metas[i]
            hits.append({
                "source": meta["source"],
                "page": int(meta["page"]),
                "score": score,
                "text": self.index.texts[i],
            })
            if len(hits) >= k:
                break
        return hits

    # -------- Optional page window (works for TF-IDF hits) --------
    def get_page_window(self, source: str, center_page: int, window: int = 1, max_chars: int = 1200) -> Tuple[str, List[int]]:
        pages: List[int] = []
        parts: List[str] = []
        for p in range(max(1, center_page - window), center_page + window + 1):
            key = (source, int(p))
            txt = self._page_map.get(key, "")
            if txt:
                pages.append(p)
                parts.append(txt)
        joined = " ".join(parts).strip()
        if len(joined) > max_chars:
            joined = joined[: max_chars - 1] + "…"
        return joined, pages
