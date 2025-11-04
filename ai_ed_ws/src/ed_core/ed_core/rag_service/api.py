# api.py
from __future__ import annotations

import os, time, re, json, math, hashlib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Body, Request
from fastapi.responses import JSONResponse
from fastapi import APIRouter

from importlib.resources import files

try:
    from importlib.metadata import version as _pkg_version
except Exception:
    _pkg_version = None

from .config import settings
from .services.llm import LLM

# ----------------------------------------------------
# (NEW) Context shim import to preserve Phi-3 behavior
# ----------------------------------------------------
# This shim can patch/wrap the LLM for special context handling.
# If it's not available, we noop cleanly.
_CONTEXT_SHIM_OK = False
try:
    # Example expected API in your shim:
    # def apply_context_shim(llm: LLM, settings) -> LLM: ...
    from .services.context_shim import apply_context_shim  # type: ignore
    _CONTEXT_SHIM_OK = True
except Exception:
    def apply_context_shim(llm, _settings):
        return llm

# ---------------------------------------
# VectorStore import with graceful fallback
# ---------------------------------------
_VECTORSTORE_OK = True
try:
    from .services.vectorstore import (
        VectorStore, ensure_dir_writable, drop_db, chroma_dir_global, db_size_bytes, gpu_info
    )
except Exception as _vs_import_err:
    _VECTORSTORE_OK = False

    def ensure_dir_writable(_p):
        try:
            Path(_p).mkdir(parents=True, exist_ok=True)
            test = Path(_p) / ".writable"
            test.write_text("ok", encoding="utf-8")
            test.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def drop_db(_p):
        try:
            p = Path(_p)
            if p.exists():
                for child in p.glob("**/*"):
                    try:
                        if child.is_file():
                            child.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def chroma_dir_global(p):
        return p

    def db_size_bytes(p):
        try:
            pth = Path(p)
            if not pth.exists():
                return 0
            return sum(f.stat().st_size for f in pth.glob("**/*") if f.is_file())
        except Exception:
            return 0

    def gpu_info():
        try:
            err = _vs_import_err
        except Exception as e:
            err = e
        return {"ok": False, "error": f"vectorstore unavailable: {err.__class__.__name__}: {err}"}

    class _StubVectorStore:
        def __init__(self, *_, **__):
            self._ok = False
            self._index_dir = str(settings.chroma_db_dir)

        def ensure_ready(self):
            pass

        def status(self) -> dict:
            return {
                "exists": False,
                "count": 0,
                "index_dir": self._index_dir,
                "size_bytes": 0,
                "error": "VectorStore disabled (missing dependency, e.g., PyMuPDF/fitz)"
            }

        def query(self, *_args, **_kwargs) -> list:
            return []

    VectorStore = _StubVectorStore  # type: ignore

# ---------------------------------------
# Safe /emb router mounting (lightweight stub by default)
# ---------------------------------------
_EMB_BACKEND = os.getenv("ED_EMB_BACKEND", "auto").lower().strip()
_EMB_NAME = os.getenv("ED_EMBED_NAME", "gte-small")
try:
    _EMB_DIM = int(os.getenv("ED_EMBED_DIM", "384"))
except Exception:
    _EMB_DIM = 384

def _stable_hash_vec(text: str, dim: int) -> List[float]:
    tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
    vec = [0.0] * max(8, dim)
    for t in tokens:
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little", signed=False)
        idx = h % dim
        contrib = ((h >> 7) & 0xFFFF) / 65535.0
        vec[idx] += 0.5 + 0.5 * contrib
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec[:dim]]

def _build_stub_emb_router() -> APIRouter:
    r = APIRouter()

    @r.get("/health")
    def _health():
        return {"ok": True, "backend": "stub", "model": _EMB_NAME, "dim": _EMB_DIM}

    @r.post("/encode")
    def _encode(payload=Body(...)):
        texts = payload.get("texts") or payload.get("input") or []
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            return {"ok": False, "error": "texts must be a list[str] or 'input'"}
        embs = [_stable_hash_vec(str(t), _EMB_DIM) for t in texts]
        return {"ok": True, "model": _EMB_NAME, "dim": _EMB_DIM, "embeddings": embs}

    return r

def _mount_emb_router(app: FastAPI) -> str:
    backend = "stub"
    if _EMB_BACKEND == "st":
        try:
            from .services.emb_router import router as _heavy_router
            app.include_router(_heavy_router, prefix="/emb")
            backend = "sentence_transformers"
        except Exception as e:
            app.state._emb_import_error = f"{e.__class__.__name__}: {e}"
            app.include_router(_build_stub_emb_router(), prefix="/emb")
            backend = "stub"
    else:
        app.include_router(_build_stub_emb_router(), prefix="/emb")
        backend = "stub"
    return backend

app = FastAPI()
_EMB_BACKEND_MOUNTED = _mount_emb_router(app)

# Always return JSON for uncaught errors
@app.exception_handler(Exception)
async def _catch_all_exceptions(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

# ----------------------------
# Load system persona
# ----------------------------
with (files("ed_core.rag_service") / "system_prompt.md").open("r", encoding="utf-8") as f:
    PERSONA = f.read().strip()

CONFIDENCE_POLICY = (
    "Answering policy:\n"
    "- If at least one relevant context line is provided, produce the best, concise answer you can from that context, "
    "cite files like [filename.pdf p.X], and add a one-line 'Limitations:' note when details are missing.\n"
    "- Only say you cannot answer when NO useful context is available and the question is outside ED’s core scope.\n"
    "- Use first-person as ED for identity/emotions/capabilities; do NOT cite docs for those.\n"
    "- Never output placeholders like '########' or generic apologies when context exists.\n"
)

_vs: VectorStore | None = None
_llm: LLM | None = None

_ED_STRICT_RAG = bool(int(os.getenv("ED_STRICT_RAG", "1")))
_ED_USE_LLM_SUMMARY = bool(int(os.getenv("ED_USE_LLM_SUMMARY", "1")))
_ED_RULEBOOK_FALLBACK = bool(int(os.getenv("ED_RULEBOOK_FALLBACK", "1")))

# ----------------------------
# Logging controls
# ----------------------------
_ED_LOG_QUERIES = bool(int(os.getenv("ED_LOG_QUERIES", "1")))
_ED_LOG_DIR = Path(os.getenv("ED_LOG_DIR", str(Path.home() / "ai_ed_ws" / "log")))
_ED_LOG_DIR.mkdir(parents=True, exist_ok=True)
_ED_LOG_PATH = _ED_LOG_DIR / "ed_api.jsonl"

def _dedup_list(seq: List[str]) -> List[str]:
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _log_query(record: Dict[str, Any]) -> None:
    if not _ED_LOG_QUERIES:
        return
    try:
        _ED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ED_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ----------------------------
# ANSI helpers (pretty TTY)
# ----------------------------
_COLORS = {
    "cyan": "\x1b[36m", "magenta": "\x1b[35m", "yellow": "\x1b[33m",
    "green": "\x1b[32m", "blue": "\x1b[34m", "reset": "\x1b[0m",
}
def _ansi(text: str, color: str = "cyan") -> str:
    c = _COLORS.get(color, _COLORS["cyan"])
    return f"{c}{text}{_COLORS['reset']}"

def _ansi_sources_list(sources: List[str]) -> List[str]:
    palette = ["cyan", "magenta", "yellow", "green", "blue"]
    return [_ansi(f"{i+1:02d} ▸ {s}", palette[i % len(palette)]) for i, s in enumerate(sources)]

# ----------------------------
# Helpers
# ----------------------------
def _ensure_vs_ready():
    global _vs
    if _vs is None:
        _vs = VectorStore()
    ensure_dir_writable(settings.chroma_db_dir)
    if getattr(settings, "rebuild_on_start", False):
        drop_db(settings.chroma_db_dir)
    _vs.ensure_ready()

def _basename_only(path_like: str | None) -> str:
    if not path_like: return "unknown"
    try:
        return Path(path_like).name
    except Exception:
        return path_like.split("/")[-1]

def _extract_text_and_meta(hit: Any) -> Tuple[str, Dict[str, Any]]:
    text = (
        getattr(hit, "page_content", None)
        or getattr(hit, "content", None)
        or (hit.get("page_content") if isinstance(hit, dict) else None)
        or (hit.get("content") if isinstance(hit, dict) else None)
        or (hit.get("text") if isinstance(hit, dict) else None)
        or ""
    )
    meta = (
        getattr(hit, "metadata", None)
        or (hit.get("metadata") if isinstance(hit, dict) else None)
        or {}
    )
    if "source" not in meta and isinstance(hit, dict) and "source" in hit:
        meta["source"] = hit["source"]
    if "page" not in meta and isinstance(hit, dict) and "page" in hit:
        meta["page"] = hit["page"]
    return str(text or ""), dict(meta or {})

def _mk_cite(meta: Dict[str, Any]) -> Tuple[str, str, int | None]:
    src = meta.get("title") or meta.get("source") or meta.get("path") or "unknown"
    src_name = _basename_only(src)
    page = meta.get("page") or meta.get("page_number") or meta.get("pageIndex")
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None
    label = f"{src_name}{f' p.{page}' if page is not None else ''}"
    return label, src_name, page

def _clean_snippet(s: str, max_len: int = 240) -> str:
    s = " ".join(s.split())
    return s if len(s) <= max_len else (s[: max_len - 1].rstrip() + "…")

# ---------- Command detection ----------
_CMD_PATTERNS = {
    "go_home": re.compile(r"\b(go\s*home|home position|return home)\b", re.I),
    "open_gripper": re.compile(r"\b(?:open|release|unlock)\b.*\bgripper\b|\bgripper\b.*\b(?:open|release|unlock)\b", re.I),
    "close_gripper": re.compile(r"\b(?:close|grab|lock)\b.*\bgripper\b|\bgripper\b.*\b(?:close|grab|lock)\b", re.I),
    "scan_scene": re.compile(r"\b(scan(?: the)? scene|look around|scan\b)\b", re.I),
}
_APRILTAG_RE = re.compile(r"\b(?:apriltag|tag)\b.*?(?:id|#)?\s*(\d+)", re.I)

def _detect_ed_cmds(q: str) -> list[dict]:
    low = q or ""
    hits: list[tuple[int, dict]] = []
    for name, rx in _CMD_PATTERNS.items():
        for m in rx.finditer(low):
            hits.append((m.start(), {"command": name, "parameters": {}}))
    for m in _APRILTAG_RE.finditer(low):
        try:
            tag_id = int(m.group(1))
            hits.append((m.start(), {"command": "pick_apriltag_ik", "parameters": {"id": tag_id}}))
        except Exception:
            pass
    hits.sort(key=lambda x: x[0])
    return [h[1] for h in hits]

def _is_pure_command(q: str, cmds: list[dict]) -> bool:
    if not cmds: return False
    if "?" in q: return False
    return len(q.strip().split()) <= 14

# ---- Intent helpers ----
_ID_PATTERNS = [
    "who are you", "who built you", "who created you",
    "origin", "which university", "supervisor", "who made you",
    "why were you built", "what are you", "who designed you"
]
_REQUIRED_ANY = ["Aston", "Jetson", "Agbor", "Edouard", "Ransome"]
_FORBIDDEN = ["Microsoft", "OpenAI", "[Company Name]", "Bing", "Azure", "ChatGPT"]

_OFFDOMAIN_PATTERNS = [
    "joke", "funny", "laugh", "love", "feelings", "feel", "girlfriend", "boyfriend",
    "can you make me money", "make us billions", "billion",
    "politics", "religion",
    "earnings", "quarter", "quarterly", "stock", "share price", "revenue", "profit", "sales",
    "nvda", "nvidia", "tesla", "apple", "google", "microsoft financials",
    "news", "latest", "today", "weather", "score", "match", "election", "president",
]

_IN_SCOPE_PATTERNS = [
    "lockout", "tagout", "loto", "osha", "control of hazardous energy", "permit to work",
    "cnc", "haas", "fanuc", "g-code", "m-code", "robot", "gripper", "apriltag", "home position",
    "lean", "5s", "kanban", "smed", "oee", "kaizen", "wip", "pull system",
    "iso 9001", "iatf", "16949", "nonconform", "audit", "process interaction map",
    "3d print", "additive", "filament", "nozzle", "extruder",
    "calibrate", "lubricate", "preventive maintenance", "pneumatic", "hydraulic", "lock pin", "limit switch",
    "ultimaker", "s7", "first layer", "first-layer", "adhesion", "z-offset", "z offset", "bed level", "build plate",
    "g54", "g55", "g56", "g57", "g58", "g59", "work coordinate", "wcs"
]

def _is_identity_query(q: str) -> bool:
    low = q.lower()
    return any(p in low for p in _ID_PATTERNS)

def _is_offdomain_query(q: str) -> bool:
    low = q.lower()
    return any(p in low for p in _OFFDOMAIN_PATTERNS)

def _is_in_scope_query(q: str) -> bool:
    low = q.lower()
    return any(p in low for p in _IN_SCOPE_PATTERNS)

def _identity_output_ok(s: str) -> bool:
    if not s or len(s.strip()) < 10: return False
    S = s.lower()
    if any(bad.lower() in S for bad in _FORBIDDEN): return False
    hits = sum(1 for req in _REQUIRED_ANY if req.lower() in S)
    return hits >= 2

def _truncate_to_two_sentences(s: str) -> str:
    parts = re.split(r'(?<=[.!?])\s+', s.strip())
    return s.strip() if len(parts) <= 2 else " ".join(parts[:2]).strip()

# ---------- Post-processing / guardrails ----------
_FORBID_BRANDS_RE = re.compile(
    r"(?ix)\b(m\s*i\s*c\s*r\s*o\s*s\s*o\s*f\s*t|o\s*p\s*e\s*n\s*a\s*i|a\s*z\s*u\s*r\s*e|b\s*i\s*n\s*g|c\s*h\s*a\s*t\s*g\s*p\s*t|g\s*p\s*t(?:-?\s*\d+)?|c\s*o\s*p\s*i\s*l\s*o\s*t|a\s*n\s*t\s*h\s*r\s*o\s*p\s*i\s*c|c\s*l\s*a\s*u\s*d\s*e|g\s*e\s*m\s*i\s*n\s*i|b\s*a\s*r\s*d|l\s*l\s*a\s*m\s*a\s*\d*)\b",
    re.IGNORECASE,
)
_BAD_OPENERS_RE = re.compile(r"^\s*(?:as\s+an?\s+ai\b|i\s*(?:'m| am)\s+an?\s+ai\b|as\s+an?\s+assistant\b)", re.I)
_GENERIC_DISCLAIMER_RE = re.compile(r"(?ix)\b(knowledge\s+cut-?off|real-?time\s+access|live\s+data|internet\s+access|browsing|language\s+model|trained\s+on\s+data|cannot\s+provide\s+financial\s+advice)\b")
_AS_AI_RE = re.compile(r"\bas\s+an?\s+ai\b", re.IGNORECASE)

def _force_ed_voice(ans: str) -> str:
    if not ans: return ans
    kept: List[str] = []
    for l in ans.splitlines():
        if _BAD_OPENERS_RE.search(l) or _GENERIC_DISCLAIMER_RE.search(l):
            continue
        kept.append(l)
    out = "\n".join(kept).strip()
    out = _AS_AI_RE.sub("as ED", out)
    out = re.sub(
        r"\b(I\s+(?:do\s+not|can't|cannot)\s+(?:access|browse).{0,60})",
        "I run fully offline on a Jetson, so I don’t access live internet.",
        out,
        flags=re.IGNORECASE,
    )
    out = _FORBID_BRANDS_RE.sub("an external provider", out)
    out = re.sub(r"(?im)^\s*for more information.*$", "", out).strip()
    return out

def _sanitize_identity_leaks(ans: str, fallback_identity: bool = False) -> str:
    if not ans: return ans
    lines = []
    for l in ans.splitlines():
        if _BAD_OPENERS_RE.search(l) or _GENERIC_DISCLAIMER_RE.search(l):
            continue
        lines.append(l)
    cleaned = "\n".join(lines).strip()
    cleaned = _FORBID_BRANDS_RE.sub("an external provider", cleaned)
    if fallback_identity and len(cleaned.strip()) < 20:
        cleaned = ("I’m ED, a shop-floor assistant built by Agbor Edouard Ransome at Aston University "
                   "(supervised by Dr. Abdullah). I run fully offline on an NVIDIA Jetson to help with safety and productivity.")
    return cleaned

# ----------------------------
# Summarizer fallbacks
# ----------------------------
def _looks_like_reference_list(ans: str) -> bool:
    if not ans: return False
    lines = [l.strip() for l in ans.splitlines() if l.strip()]
    if not lines: return False
    ref_hits = 0
    for l in lines[:6]:
        if re.search(r"(OSHA|ANSI|ISO|IATF|Manual|Publication|eTool|Standards|References|Sources)", l, re.I): ref_hits += 1
        if re.search(r"\b(p\.\d+)\b", l): ref_hits += 1
        if re.search(r"(?:https?://|\bCFR\s*1910)", l, re.I): ref_hits += 1
        if re.match(r"^\d+[\.\)]\s", l): ref_hits += 1
    return ref_hits >= 2

def _looks_like_multi_cite_dump(ans: str) -> bool:
    if not ans: return False
    cites = re.findall(r"\[[^\]]+\.pdf(?:\s+p\.\d+)?\]", ans, flags=re.IGNORECASE)
    return len(cites) >= 6

try:
    from .summarizer import (
        cited_sentences_from_ctx,
        looks_like_ref_dump,
        ensure_numbered_with_cites,
    )
except Exception:
    def cited_sentences_from_ctx(ctx_lines, max_sentences=4):
        return _synthesize_sentences_from_ctx(ctx_lines, max_sentences=max_sentences)
    def looks_like_ref_dump(text: str) -> bool:
        return _looks_like_reference_list(text) or _looks_like_multi_cite_dump(text)
    def ensure_numbered_with_cites(raw: str, ctx_lines: List[str]) -> str:
        return _ensure_procedure_numbered_with_cites(raw, ctx_lines)

def _ensure_procedure_numbered_with_cites(ans: str, ctx_lines: List[str]) -> str:
    cites = _ctx_citations_from_ctx_lines(ctx_lines)
    if not ans or _looks_like_reference_list(ans):
        steps = []
        for i, line in enumerate(ctx_lines[:8], 1):
            m = re.match(r"^- \[([^\]]+)\]\s*(.*)$", line)
            cite = m.group(1) if m else (cites[i-1] if i-1 < len(cites) else "")
            txt = (m.group(2) if m else line.lstrip("- ").strip())
            if re.search(r"\b(references?|publication|annex|appendix|table\s+\d+|29\s*cfr|cfr\s*1910)\b", txt, re.I):
                continue
            first = re.split(r'(?<=[.!?])\s+', txt.strip())[0].rstrip(".")
            steps.append(f"{i}. {first}" + (f" [{cite}]" if cite else ""))
        if steps:
            return "\n".join(steps)
        return ans

    out_lines: List[str] = []
    num = 1
    for raw in ans.splitlines():
        l = raw.strip()
        if not l:
            continue
        if re.match(r"^\d+[\.\)]\s", l):
            step_txt = re.sub(r"^\d+[\.\)]\s*", "", l).strip()
        elif re.match(r"^[\-\*]\s", l):
            step_txt = re.sub(r"^[\-\*]\s*", "", l).strip()
        else:
            if out_lines:
                out_lines[-1] = out_lines[-1] + " " + l
            else:
                out_lines.append(f"{num}. {l}")
                num += 1
            continue

        has_cite = bool(re.search(r"\[[^\]]+\.pdf(?:\s+p\.\d+)?\]", step_txt, re.I))
        cite = cites[(num-1) % len(cites)] if cites else ""
        line_norm = f"{num}. {step_txt}" + ("" if has_cite or not cite else f" [{cite}]")
        out_lines.append(line_norm)
        num += 1

    out_lines = out_lines[:10]
    if cites and not any("[" in l and "]" in l for l in out_lines):
        out_lines.append("Limitations: citations were not available in the retrieved context.")
    return "\n".join(out_lines) if out_lines else ans

def _synthesize_sentences_from_ctx(ctx_lines: List[str], max_sentences: int = 4) -> str:
    sentences: List[str] = []
    for line in ctx_lines:
        m = re.match(r"^- \[([^\]]+)\]\s*(.*)$", line)
        cite = m.group(1) if m else ""
        body = (m.group(2) if m else line.lstrip("- ").strip())
        txt = body.strip()
        txt = re.sub(r"^\s*(?:(?:chapter|section)\s+\d+(?:\.\d+){0,3}\s*[:\-]?\s*|\d+(?:\.\d+){0,3}\s*[)\.]?\s+)", "", txt, flags=re.I)
        if len(re.findall(r"[A-Za-z]", txt)) < 6: continue
        sent = re.split(r'(?<=[.!?])\s+', txt)[0]
        sent = " ".join(sent.split())
        if not sent: continue
        if len(sent) > 220:
            sent = sent[:220].rsplit(" ", 1)[0] + "…"
        sentences.append(sent + (f" [{cite}]" if cite else ""))
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return "I don’t have enough grounded context to answer that. Limitations: no relevant local excerpts were retrieved."
    return " ".join(sentences)

_TWO_SENTENCE_FLAG_RE = re.compile(r"\b(two|2)\s+sentences?\b", re.I)
_INLINE_CITE_FLAG_RE = re.compile(r"\binline\s+cite|\bone\s+inline\s+cite\s+per\s+sentence\b", re.I)
def _normalize_question(q: str) -> str:
    if not q: return q
    q = _AS_AI_RE.sub("as ED", q)
    q = re.sub(r"^\s*(?:as\s+an?\s+assistant\b).*?\b,?\s*", "", q, flags=re.I)
    return q.strip()
def _wants_two_sentence_cites(q: str) -> bool:
    return bool(_TWO_SENTENCE_FLAG_RE.search(q) or _INLINE_CITE_FLAG_RE.search(q))

def _clip_trailing_fragments(text: str, max_words: int = 70) -> str:
    if not text: return text
    words = text.strip().split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip() + "…"
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return text.strip()
    last = lines[-1].strip()
    if last.endswith(":") or last.endswith("[") or last.endswith("("):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _ensure_exact_two_sentences(text: str) -> str:
    parts = re.split(r'(?<=[.!?])\s+', " ".join(text.split()))
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 2:
        a, b = parts
        if not a.endswith((".", "!", "?")): a += "."
        if not b.endswith((".", "!", "?")): b += "."
        return a + " " + b
    if not parts:
        return ""
    merged = " ".join(parts)
    parts = re.split(r'(?<=[.!?])\s+', merged)
    parts = [p.strip() for p in parts if p.strip()][:2]
    if not parts: return ""
    if len(parts) == 1: parts.append("")
    a, b = parts
    if not a.endswith((".", "!", "?")): a += "."
    if b and not b.endswith((".", "!", "?")): b += "."
    return (a + " " + b).strip()

# ---------- Domain-aware source filtering ----------
_BAD_SOURCES_BY_DOMAIN = {
    "cnc": {
        "deny": {
            "artificial_intelligence_tutorial.pdf",
            "Lean-Manufacturing-Tools-Techniques-and-How-To-Use-Them.pdf",
        },
        "require_any": {"haas", "fanuc", "operator", "manual", "lathe", "mill"}
    },
    "3dp": {
        "deny": {
            "artificial_intelligence_tutorial.pdf",
            "english_lathe_interactive_manual_print_version_2023.pdf",
        },
        "require_any": {"ultimaker", "s7", "print", "3d"}
    },
    "loto": {
        "deny": {
            "artificial_intelligence_tutorial.pdf",
        },
        "require_any": {"osha", "1910.147", "lockout", "tagout"}
    }
}
def _allowed_source_for_domain(label_or_path: str, domain: str) -> bool:
    domain = (domain or "general").lower()
    s = (label_or_path or "").lower()
    rules = _BAD_SOURCES_BY_DOMAIN.get(domain, {})
    for bad in rules.get("deny", set()):
        if bad.lower() in s:
            return False
    req = rules.get("require_any", set())
    if req and not any(tok in s for tok in req):
        return False
    return True

# ---------- Question-aware extraction & scoring ----------
_STOPWORDS = set("a an and are as at be but by for from has have if in into is it its of on or such that the their then there these this to was were will with your you".split())

def _focus_terms(q: str, extra: List[str] | None = None, max_terms: int = 6) -> List[str]:
    words = re.findall(r"[A-Za-z0-9\-]+", (q or ""))
    words = [w.lower() for w in words if w and w.lower() not in _STOPWORDS and len(w) > 2]
    prof = _domain_profile(q)
    pool = words + [k.lower() for k in prof.get("keywords", [])] + (extra or [])
    seen, out = set(), []
    for w in pool:
        if w not in seen:
            seen.add(w); out.append(w)
        if len(out) >= max_terms:
            break
    return out

def _split_into_sentences(text: str) -> List[str]:
    if not text: return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(])', " ".join(text.split()))
    return [p.strip() for p in parts if len(re.findall(r"[A-Za-z]", p)) >= 4]

def _score_sentence_for_question(sent: str, focus: List[str]) -> float:
    s = sent.lower(); score = 0.0
    for f in focus:
        if f in s: score += 1.0
    if re.search(r"\b(define|means?|used\s+to|sets?|controls?|verif(y|ies)|ensure|warning|caution|safe(?:ly)?)\b", s): score += 0.6
    if re.search(r"\b(table|figure|publication|references|standards|etool|annex|appendix)\b", s): score -= 0.8
    if len(s) < 30: score -= 0.3
    return score

def _score_step_line(line: str) -> float:
    s = line.strip()
    if not s: return -1.0
    S = s.lower(); score = 0.0
    if re.match(r"^(?:\d+[\.\)]\s+|step\s*\d+\s*[:\-]\s+)", s, re.I): score += 0.6
    if re.match(r"^(verify|ensure|apply|isolate|lock|bleed|vent|block|press|select|set|check|clean|level|adjust|wear|remove|install|calibrate)\b", S): score += 0.8
    if re.match(r"^\d+(\.\d+){1,3}\b", s): score -= 0.7
    if re.search(r"\b(references|publication|eTool|annex|appendix|table\s+\d+)\b", S): score -= 0.7
    if 8 <= len(s.split()) <= 28: score += 0.2
    return score

def _compose_answer_from_ctx(q: str, ctx_lines: List[str], max_sentences: int = 4) -> str:
    if not ctx_lines: return ""
    focus = _focus_terms(q)
    items = []
    for line in ctx_lines:
        m = re.match(r"^- \[([^\]]+)\]\s*(.*)$", line)
        if not m: continue
        cite, body = m.group(1), m.group(2)
        items.append((cite, body))
    scored = sorted(items, key=lambda x: _score_sentence_for_question(x[1], focus), reverse=True)
    picked = []
    seen_claims = set()
    for cite, body in scored:
        sents = _split_into_sentences(body) or [body.strip()]
        if not sents: continue
        s = sents[0]
        key = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()[:80]
        if key in seen_claims: continue
        seen_claims.add(key)
        picked.append((cite, _clean_snippet(s, 220)))
        if len(picked) >= max(2, min(5, int(max_sentences))): break
    if not picked: return ""
    lead = picked[0][1]
    out_sents = [lead.rstrip(".") + f" [{picked[0][0]}]"]
    for cite, s in picked[1:]:
        out_sents.append(s.rstrip(".") + f" [{cite}]")
    return " ".join(out_sents)

# ----------------------------
# Domain profile / rerank
# ----------------------------
def _domain_profile(q: str) -> dict:
    low = q.lower()
    if any(k in low for k in ["lockout", "tagout", "loto", "1910.147", "osha"]):
        return {"name":"loto","keywords":["lockout","tagout","loto","1910.147","osha","energy control","verify","isolate","stored energy"],
                "prefer_sources":["osha","lockout","tagout","safety"],"expansions":[q,q+" lockout tagout",q+" LOTO",
                "OSHA 1910.147 lockout tagout steps","control of hazardous energy lockout tagout procedure"],
                "proc_threshold":0.08,"win_pages":2}
    if any(k in low for k in ["cnc","haas","fanuc","g-code","m-code","home position","g54","g55","g56","g57","g58","g59","work coordinate","wcs"]):
        return {"name":"cnc","keywords":["cnc","haas","fanuc","g-code","m-code","home position","offset","alarm","parameter","g54","g55","g56","g57","g58","g59","work coordinate","wcs"],
                "prefer_sources":["haas","fanuc","operator","manual"],"expansions":[q, q+" haas", q+" fanuc", q+" operator manual", q+" work offset"],
                "proc_threshold":0.07,"win_pages":1}
    if any(k in low for k in ["kanban","5s","lean","kaizen","smed","oee","wip"]):
        return {"name":"lean","keywords":["lean","5s","kaizen","kanban","smed","oee","wip","pull system"],
                "prefer_sources":["kanban","lean","5s","guide"],"expansions":[q,q+" steps",q+" checklist"],
                "proc_threshold":0.06,"win_pages":1}
    if any(k in low for k in ["iso 9001","iatf","16949","audit","nonconform"]):
        return {"name":"iso","keywords":["iso 9001","iatf 16949","audit","process map","nonconformity"],
                "prefer_sources":["iso","iatf"],"expansions":[q,q+" requirements",q+" process"],
                "proc_threshold":0.06,"win_pages":1}
    if any(k in low for k in ["3d print","additive","filament","nozzle","extruder","ultimaker","s7","first layer","first-layer","adhesion","z-offset","z offset","bed level","build plate"]):
        return {"name":"3dp","keywords":["3d print","additive","filament","nozzle","extruder","bed","adhesion","first layer","z-offset","ultimaker","s7","build plate","bed level","first-layer"],
                "prefer_sources":["ultimaker","print","3d","additive","s7"],
                "expansions":[q, q+" troubleshooting", q+" best practices", q+" first layer", q+" adhesion", q+" ultimaker", q+" ultimaker s7 manual", q+" z-offset"],
                "proc_threshold":0.06,"win_pages":1}
    return {"name":"general","keywords":[],"prefer_sources":[],"expansions":[q],"proc_threshold":0.06,"win_pages":1}

def _collect_hits_with_rerank(q: str, k: int, thr: float, procedure_mode: bool) -> list:
    prof = _domain_profile(q)
    prefer = [p.lower() for p in prof["prefer_sources"]]
    keyset = set(k.lower() for k in prof["keywords"])
    base_thr = max(thr, prof.get("proc_threshold", thr)) if procedure_mode else thr
    pool: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for alt in prof["expansions"]:
        for h in _vs.query(alt, k=max(k, 6), threshold=base_thr):
            src = h.get("source") or "unknown"
            page_raw = h.get("page")
            try:
                page_key = int(page_raw) if page_raw is not None else -1
            except Exception:
                page_key = -1
            key = (src, page_key)
            if key not in pool or h["score"] > pool[key]["score"]:
                pool[key] = h
    hits = list(pool.values())
    if not hits: return []

    cnc_g5x_re = re.compile(r"\bG5[4-9]\b|\bwork\s+coordinate\b|\bWCS\b", re.I)
    first_layer_re = re.compile(r"\bfirst[- ]?layer\b|\badhesion\b|\bz[- ]?offset\b", re.I)

    def _boost(h):
        src = (h.get("source") or "").lower()
        _txt, _meta = _extract_text_and_meta(h)
        txt = " ".join((_txt or "").lower().split())
        b = 0.0
        if any(p in src for p in prefer): b += 0.12
        kw_hits = sum(1 for kw in keyset if kw and (kw in txt or kw in src))
        b += min(0.10, 0.03 * kw_hits)
        if procedure_mode and re.search(r"(\n\s*\d+\.\s|\bstep\s+\d+|\bprocedure\b)", txt): b += 0.06
        if prof["name"] == "cnc" and cnc_g5x_re.search(txt or ""): b += 0.10
        if prof["name"] == "3dp" and first_layer_re.search(txt or ""): b += 0.10
        return h["score"] + b

    hits.sort(key=_boost, reverse=True)
    domain = prof["name"]
    hits = [h for h in hits if _allowed_source_for_domain((h.get("source") or ""), domain)]
    return hits[:k]

def _short_topic_phrase(q: str) -> str:
    words = re.findall(r"[A-Za-z0-9\-\/\.']+", q.strip())
    return " ".join(words[:8])

def _suggest_followups(area_name: str) -> list[str]:
    if area_name == "loto": return ["When is LOTO mandatory?", "Show an example LOTO checklist."]
    if area_name == "cnc":  return ["What does G54 vs G92 do?", "How do I safely home the robot?"]
    if area_name == "lean": return ["What are the 5S steps?", "How to set WIP limits in Kanban?"]
    if area_name == "iso":  return ["ISO 9001 vs IATF 16949 differences?", "What is a process interaction map?"]
    if area_name == "3dp":  return ["Best first-layer checklist?", "PLA vs ABS nozzle temps?"]
    return ["Ask about LOTO, CNC/robots, 5S/Kanban, ISO/IATF, or maintenance."]

# ----------------------------
# RULEBOOK
# ----------------------------
RuleHit = Dict[str, Any]
def _rb_loto_steps(_: re.Match) -> RuleHit:
    ans = (
        "Lockout/Tagout – core steps:\n"
        "1) Notify affected personnel and review the energy-control plan.\n"
        "2) Shut down the machine using normal controls.\n"
        "3) Isolate all energy sources (electrical, pneumatic, hydraulic, mechanical, thermal, etc.).\n"
        "4) Apply locks and tags to each isolation point; one person, one lock.\n"
        "5) Dissipate or restrain stored energy (bleed, vent, block, discharge, release springs).\n"
        "6) Verify zero energy: try-start and test with appropriate instruments.\n"
        "7) Perform the work.\n"
        "8) Restore to service: clear tools/people, remove locks/tags in reverse order, re-energize, and notify personnel.\n"
        "Note: Follow your site’s written procedure and authorized person training."
    )
    return {"answer": ans, "rule": "loto_steps"}
def _rb_oee(_: re.Match) -> RuleHit:
    ans = (
        "OEE (Overall Equipment Effectiveness) = Availability × Performance × Quality.\n"
        "• Availability = Run Time / Planned Production Time.\n"
        "• Performance = (Ideal Cycle Time × Total Count) / Run Time.\n"
        "• Quality = Good Count / Total Count.\n"
        "Example: If Availability 0.90, Performance 0.95, Quality 0.98 → OEE = 0.90×0.95×0.98 ≈ 83.7%."
    )
    return {"answer": ans, "rule": "oee_basics"}
def _rb_kanban(_: re.Match) -> RuleHit:
    ans = (
        "Kanban is a pull system that limits WIP and signals work based on demand. "
        "Use pull when demand is variable, changeovers are modest, and you can buffer small inventory at constraint points. "
        "Key practices: visualize work, set WIP limits, manage flow, make policies explicit, and improve iteratively."
    )
    return {"answer": ans, "rule": "kanban_pull"}
def _rb_iso_iatf(_: re.Match) -> RuleHit:
    ans = (
        "ISO 9001 is a general quality-management standard for any industry. "
        "IATF 16949 builds on ISO 9001 with automotive-specific requirements (e.g., customer-specific requirements, "
        "APQP/PPAP, traceability, defect prevention, risk-based thinking across the supply chain)."
    )
    return {"answer": ans, "rule": "iso_vs_iatf"}
def _rb_g54(_: re.Match) -> RuleHit:
    ans = (
        "G54 is a work coordinate system (WCS) offset on CNC machines (G54–G59). "
        "It defines the part zero relative to machine zero so programs reference the correct origin. "
        "Typical use: touch off/set work offset, verify with a safe dry run, then run at feed.\n"
        "Caution: Always confirm the active WCS on the control before cycle start."
    )
    return {"answer": ans, "rule": "cnc_g54"}
def _rb_3dp_first_layer(_: re.Match) -> RuleHit:
    ans = (
        "First-layer calibration:\n"
        "1) Clean bed; use correct surface prep.\n"
        "2) Level bed and set Z-offset so skirt lines are slightly squished with no gaps.\n"
        "3) Set temps: nozzle/bed per filament; avoid drafts.\n"
        "4) Start slow (15–25 mm/s) with ~100% flow for first layer.\n"
        "5) Verify extrusion width is consistent; adjust Z-offset ±0.02 mm if lines are too round (too high) or ridged (too low)."
    )
    return {"answer": ans, "rule": "3dp_first_layer"}
def _rb_robot_home(_: re.Match) -> RuleHit:
    ans = (
        "Robot safety check before homing:\n"
        "1) Area clear: no people or obstacles in the robot envelope.\n"
        "2) Guards/doors closed; interlocks functional; E-stop reset.\n"
        "3) Correct mode selected (Teach/Manual for close-range, low speed).\n"
        "4) Tooling secure; cables/airlines clear.\n"
        "5) Home at reduced speed; be ready to stop; verify axes references complete."
    )
    return {"answer": ans, "rule": "robot_safety_home"}
def _rb_5s(_: re.Match) -> RuleHit:
    ans = (
        "5S audit quicklist:\n"
        "• Sort: unnecessary items removed, red-tagged.\n"
        "• Set in Order: locations labeled; tools shadowed; point-of-use storage.\n"
        "• Shine: clean floors/machines; leaks addressed; cleaning schedule posted.\n"
        "• Standardize: visual standards, checklists, photos of ‘good’.\n"
        "• Sustain: audits posted; owners assigned; trends tracked."
    )
    return {"answer": ans, "rule": "5s_audit"}
def _rb_cnc_startup(_: re.Match) -> RuleHit:
    ans = (
        "CNC startup after maintenance (generic):\n"
        "1) Visual checks: covers on, guards closed, fluids/air OK, no tools left in enclosure.\n"
        "2) Power on; clear alarms; reference/home all axes.\n"
        "3) Verify lubrication/air pressure; run warm-up cycle if required.\n"
        "4) Load tool offsets and work offset (e.g., G54); check spindle/tool changer dry.\n"
        "5) Dry run/graphics first; then single-block and feed-hold nearby for first part."
    )
    return {"answer": ans, "rule": "cnc_startup_after_maintenance"}

_RULES: List[Tuple[re.Pattern, Any]] = [
    (re.compile(r"\block\s*out\s*/?\s*tag\s*out|\bloto\b", re.I), _rb_loto_steps),
    (re.compile(r"\boee\b|overall\s+equipment\s+effectiveness", re.I), _rb_oee),
    (re.compile(r"\bkanban\b|\bpull\s+system\b", re.I), _rb_kanban),
    (re.compile(r"\biso\s*9001\b.*\b(iatf|16949)\b|\biatf\b|\b16949\b", re.I), _rb_iso_iatf),
    (re.compile(r"\bg54\b|\bwork\s+coordinate\b", re.I), _rb_g54),
    (re.compile(r"(first\s*layer|z[- ]?offset|bed\s*level|adhesion)", re.I), _rb_3dp_first_layer),
    (re.compile(r"robot.*(safety|check).*(homing|home)|before\s+homing", re.I), _rb_robot_home),
    (re.compile(r"\b5s\b.*audit|\bfive\s*s\b", re.I), _rb_5s),
    (re.compile(r"(startup|start\s*up).*\bcnc\b|after\s+maintenance.*\bcnc\b", re.I), _rb_cnc_startup),
]
def _rulebook_match(q: str) -> Optional[RuleHit]:
    for rx, fn in _RULES:
        m = rx.search(q)
        if m:
            try:
                return fn(m)
            except Exception:
                continue
    return None

# ----------------------------
# Templated short answers
# ----------------------------
def _templated_short_answer(q: str) -> Optional[str]:
    S = q.lower()
    if "g54" in S:
        return ("G54 is a work coordinate offset that sets part zero relative to machine zero so the program’s X/Y/Z "
                "positions reference the correct origin. To use it safely, set the offset with your probe or touch-off, "
                "confirm G54 is active on the control, and dry-run above the part before cutting.")
    if "ultimaker" in S and ("first layer" in S or "first-layer" in S):
        return ("Clean the plate and apply the right surface prep, re-level the bed, and set Z-offset so the first lines "
                "are slightly squished without gaps. Use recommended nozzle/bed temps for the filament and slow the first "
                "layer so it sticks evenly across the plate.")
    return None

# ----------------------------
# Timed LLM wrapper
# ----------------------------
def _llm_generate_timed(system: str, user: str, *, max_tokens: int, temp: float) -> Tuple[str, float]:
    t0 = time.time()
    out = _llm.generate(system=system, user=user, max_tokens=max_tokens, temp=temp).strip()
    return out, round(time.time() - t0, 3)

# ----------------------------
# Dynamic identity & helpers (timed)
# ----------------------------
def _dynamic_identity_answer(session_id: str) -> Tuple[str, float]:
    system_base = (
        PERSONA
        + "\nYou are answering an identity/origin question. Be truthful and concise.\n"
        + "Do not mention an external provider or any other company names.\n"
        + "Do not use placeholders like [Company Name].\n"
        + "Tone: practical, shop-floor friendly. 1–2 short sentences, first-person as ED.\n"
        + "Do not cite documents."
    )
    pins = (
        "Facts you MUST use (paraphrase naturally):\n"
        "- Creator: AGBOR Edouard Ransome\n"
        "- University: Aston University\n"
        "- Supervisor: Dr. Abdullah\n"
        "- Platform: NVIDIA Jetson (fully offline)\n"
        "If a fact is unknown, omit it—do not replace with anything else."
    )
    out1, g1 = _llm_generate_timed(
        system_base + "\n\n" + pins,
        "Introduce yourself to a shop-floor operator who asked about your origin and purpose.",
        max_tokens=70, temp=0.3
    )
    out1 = _truncate_to_two_sentences(out1)
    if _identity_output_ok(out1):
        return out1, g1
    stricter = system_base + "\nAnswer using EXACTLY the facts above in your own words. Never introduce other organizations or brands."
    out2, g2 = _llm_generate_timed(
        stricter + "\n\n" + pins,
        "Give a one or two sentence introduction.",
        max_tokens=70, temp=0.2
    )
    out2 = _truncate_to_two_sentences(out2)
    if _identity_output_ok(out2):
        return out2, g2
    return ("I’m ED, a shop-floor assistant built by Agbor Edouard Ransome at Aston University "
            "(supervised by Dr. Abdullah). I run fully offline on an NVIDIA Jetson to help with safety and productivity."), 0.0

def _max_tokens_for_budget(t0: float, default_max: int) -> int:
    timeout_s = float(os.getenv("ED_TIMEOUT_S", "10"))
    elapsed = time.time() - t0
    budget = max(1.0, timeout_s - elapsed)
    est_tps = 18.0
    cap = int(est_tps * budget * 0.8)
    return max(48, min(default_max, cap))

def _is_procedure_query(q: str) -> bool:
    low = q.lower()
    keys = ["steps", "procedure", "how do i", "how to", "checklist", "setup", "set up",
            "replace", "install", "calibrate", "startup", "start up", "shutdown", "reset", "view"]
    return any(k in low for k in keys)

_CITE_IN_ANSWER_RE = re.compile(r"\[([^\[\]]+?\.pdf)(?:\s+p\.(\d+))?\]", re.I)
def _citations_from_answer(ans: str) -> tuple[list[str], list[dict]]:
    if not ans: return [], []
    seen = set(); sources_out: list[str] = []; cites_out: list[dict] = []
    for m in _CITE_IN_ANSWER_RE.finditer(ans):
        file_name = Path(m.group(1)).name
        page = int(m.group(2)) if m.group(2) else None
        label = f"{file_name}{f' p.{page}' if page is not None else ''}"
        if label not in seen:
            seen.add(label); sources_out.append(label); cites_out.append({"source": file_name, "page": page})
    return sources_out, cites_out

_ROLE_TOKEN_RE = re.compile(r'^\s*(?:\[?\s*assistant\s*\]?:?>?|<\|assistant\|>)\s*', re.IGNORECASE)
def _strip_role_tokens(s: str) -> str:
    return "\n".join(_ROLE_TOKEN_RE.sub("", ln) for ln in (s or "").splitlines())
def _paragraphs(text: str) -> List[str]:
    if not text: return []
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    if blocks: return blocks
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out, buf = [], []
    for s in sents:
        if not s.strip(): continue
        buf.append(s)
        if len(buf) >= 2:
            out.append(" ".join(buf)); buf = []
    if buf: out.append(" ".join(buf))
    return out

# ----------------------------
# Two-sentence citation enforcer
# ----------------------------
def _enforce_two_sentence_cites(ans: str, ctx_lines: List[str]) -> str:
    """Ensure exactly one bracket citation per sentence using ctx labels if missing."""
    if not ans: return ans
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', ans.strip()) if s.strip()]
    if len(sentences) != 2:
        return _ensure_exact_two_sentences(ans)
    ctx_cites = _ctx_citations_from_ctx_lines(ctx_lines)
    if not ctx_cites:
        return _ensure_exact_two_sentences(ans)
    fixed = []
    for i, s in enumerate(sentences):
        if "[" in s and "]" in s:
            # already has some cite; keep it but avoid multiple cites
            # keep only last [...] group
            s2 = re.sub(r"\s*\[[^\]]+\](?!.*\[[^\]]+\])", lambda m: m.group(0), s)  # keep last
            s2 = re.sub(r"\s*\[[^\]]+\](?=.*\[[^\]]+\])", "", s2)  # drop earlier cites
            fixed.append(s2.strip())
        else:
            cite = ctx_cites[min(i, len(ctx_cites)-1)]
            fixed.append((s.rstrip(".!?") + f" [{cite}].").replace("..", "."))
    out = " ".join(fixed)
    # ensure exactly two sentences
    return _ensure_exact_two_sentences(out)

# -------- LLM authored, cited answer (Phi-3) (timed) --------
def _llm_cited_answer(
    q: str,
    ctx_lines: List[str],
    *,
    two_sentence: bool = False,
    procedure_mode: bool = False,
    t0: float | None = None,
) -> Tuple[str, float]:
    ctx_body = "\n".join(ctx_lines) if ctx_lines else "(no local context available)"
    system = (
        PERSONA
        + "\nYou are ED. Keep answers human and shop-floor practical.\n"
        + "No headings. No reference lists. Use inline bracket citations like [file.pdf p.X] tied to claims you pull from context.\n"
    )

    if procedure_mode:
        user = (
            f"Question: {q}\n\n"
            f"Context (each line begins with a cite label in brackets):\n{ctx_body}\n\n"
            "Write a short, numbered procedure (3–8 steps). Each step must be actionable and end with one inline citation "
            "matching the labels above (e.g., [file.pdf p.X]). If a step cannot be confirmed from the context, "
            "add a final 'Limitations:' line. Keep it tight."
        )
        max_tokens = _max_tokens_for_budget(t0 or time.time(), getattr(settings, 'query_max_tokens', 192))
        temp = 0.2
    else:
        if two_sentence:
            user = (
                f"Question: {q}\n\n"
                f"Context (each line begins with a cite label in brackets):\n{ctx_body}\n\n"
                "Write EXACTLY two natural sentences, first-person as ED. "
                "Each sentence must end with exactly one bracket citation drawn from the labels in Context "
                "(e.g., [file.pdf p.X]). Do not add extra sentences, lists, or headings."
            )
            max_tokens = _max_tokens_for_budget(t0 or time.time(), getattr(settings, 'query_max_tokens', 192))
            temp = getattr(settings, "query_temp", 0.4)
        else:
            user = (
                f"Question: {q}\n\n"
                f"Context (each line begins with a cite label in brackets):\n{ctx_body}\n\n"
                "Write 2–4 SHORT PARAGRAPHS in first person as ED that synthesize the context into a clear explanation. "
                "Integrate ideas; do not list snippets. Avoid headings and bullet points. "
                "Include at least ONE inline bracket citation in each paragraph, using labels from the Context (e.g., [file.pdf p.X]). "
                "If some important detail is missing, end the last paragraph with a one-line 'Limitations:' note."
            )
            max_tokens = _max_tokens_for_budget(t0 or time.time(), max(getattr(settings, 'query_max_tokens', 192), 280))
            temp = getattr(settings, "query_temp", 0.4)

    out, gen_s = _llm_generate_timed(system, user, max_tokens=max_tokens, temp=temp)
    try:
        if getattr(LLM, "is_fallback", None) and LLM.is_fallback(out):
            return "", gen_s
    except Exception:
        pass
    if out.lower().startswith(("i’m not confident", "i am not confident", "i'm not confident")):
        return "", gen_s

    def _violates_two_sentence_rules(txt: str) -> bool:
        if not two_sentence: return False
        parts = re.split(r'(?<=[.!?])\s+', " ".join(txt.split()))
        parts = [p for p in parts if p.strip()]
        if len(parts) != 2: return True
        return _looks_like_reference_list(txt) or _looks_like_multi_cite_dump(txt)

    if _violates_two_sentence_rules(out):
        stricter_user = user + "\n\nSTRICT: Exactly two sentences. No lists. One bracket citation per sentence. No extra text."
        out2, gen_s2 = _llm_generate_timed(system, stricter_user, max_tokens=max_tokens, temp=0.2)
        if not _violates_two_sentence_rules(out2):
            out, gen_s = out2, gen_s + gen_s2

    out = _force_ed_voice(_sanitize_identity_leaks(out))
    out = _strip_role_tokens(out)
    out = re.sub(r"(?im)^(?:chapter|section)\s+\d+(?:\.\d+){0,3}.*$", "", out).strip()
    out = _clip_trailing_fragments(out)

    if two_sentence:
        out = _ensure_exact_two_sentences(out)
        out = _enforce_two_sentence_cites(out, ctx_lines)
    else:
        paras = _paragraphs(out)
        ctx_cites = _ctx_citations_from_ctx_lines(ctx_lines)
        if ctx_cites and paras:
            fixed = []
            for i, p in enumerate(paras):
                if "[" not in p or "]" not in p:
                    cite = ctx_cites[min(i, len(ctx_cites)-1)]
                    p = p.rstrip() + f" [{cite}]"
                fixed.append(p)
            out = "\n".join(fixed)

    return out, gen_s

# ----------------------------
# Context builders
# ----------------------------
_BOILERPLATE_RE = re.compile(r"^(?:for more information|see also|references?:|publication|annex|appendix)\b", re.I)

def _ctx_citations_from_ctx_lines(ctx_lines: List[str]) -> List[str]:
    cites: List[str] = []
    for line in ctx_lines:
        m = re.match(r"^- \[([^\]]+)\]", line)
        if m:
            cites.append(m.group(1))
    return cites

def _pick_best_sentences(block: str, max_chars: int = 260) -> str:
    sents = _split_into_sentences(block) or [block.strip()]
    sents = [s for s in sents if not _BOILERPLATE_RE.search(s)]
    if not sents:
        return ""
    out = []
    for s in sents:
        out.append(s.strip())
        joined = " ".join(out)
        if len(joined) >= max_chars:
            break
    return _clean_snippet(" ".join(out), max_chars)

def _build_nonproc_context(hits: List[dict], max_ctx_chars: int = 1200, window_pages: int = 0, q: str = "") -> Tuple[List[str], List[str], List[dict]]:
    """Builds prose context lines: '- [file.pdf p.X] sentence(s)'"""
    lines: List[str] = []
    sources: List[str] = []
    cites: List[dict] = []
    used_keys = set()

    for h in hits:
        txt, meta = _extract_text_and_meta(h)
        if not txt: continue
        label, src_name, page = _mk_cite(meta)
        key = (src_name, page if page is not None else -1)
        if key in used_keys:
            continue
        used_keys.add(key)
        picked = _pick_best_sentences(txt, max_chars=280)
        if not picked:
            continue
        line = f"- [{label}] {picked}"
        lines.append(line)
        sources.append(label)
        cites.append({"source": src_name, "page": page})
        if sum(len(l) for l in lines) >= max_ctx_chars:
            break

    return lines, _dedup_list(sources), cites

def _extract_candidate_steps(text: str) -> List[str]:
    # Heuristic extraction of step-like lines
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates: List[str] = []
    for l in raw_lines:
        if re.match(r"^\d+[\.\)]\s+", l) or re.match(r"^\s*step\s*\d+\s*[:\-]\s+", l, re.I) or re.match(r"^[\-\*]\s+", l):
            l2 = re.sub(r"^[\-\*\d\)\. ]+\s*", "", l).strip()
            candidates.append(l2)
        else:
            # short imperative sentences
            if re.match(r"^(verify|ensure|apply|isolate|lock|bleed|vent|block|press|select|set|check|clean|level|adjust|wear|remove|install|calibrate)\b", l, re.I):
                candidates.append(l)
    return candidates

def _build_procedure_context(hits: List[dict], total_ctx_chars: int = 1200, window_pages: int = 2, max_steps: int = 6, q: str = "") -> Tuple[List[str], List[str], List[dict]]:
    """Builds procedure-oriented context lines preferring step-like snippets."""
    ctx_lines: List[str] = []
    sources: List[str] = []
    cites: List[dict] = []
    used_keys = set()

    for h in hits:
        txt, meta = _extract_text_and_meta(h)
        if not txt: continue
        label, src_name, page = _mk_cite(meta)
        key = (src_name, page if page is not None else -1)
        if key in used_keys:
            continue
        used_keys.add(key)

        # pull candidate steps, score them
        cands = _extract_candidate_steps(txt)
        if not cands:
            # fallback to a good sentence if no step pattern found
            cands = [_pick_best_sentences(txt, max_chars=200)]

        scored = sorted(((l, _score_step_line(l)) for l in cands if l), key=lambda x: x[1], reverse=True)
        take = [l for (l, sc) in scored[:2] if sc > -0.2] or [cands[0]]

        for t in take:
            snippet = _clean_snippet(t, 200)
            line = f"- [{label}] {snippet}"
            ctx_lines.append(line)
            sources.append(label)
            cites.append({"source": src_name, "page": page})
            if len(ctx_lines) >= max_steps:
                break
        if len(ctx_lines) >= max_steps:
            break
        if sum(len(l) for l in ctx_lines) >= total_ctx_chars:
            break

    return ctx_lines, _dedup_list(sources), cites

# ----------------------------
# Lifecycle
# ----------------------------
@app.on_event("startup")
def _on_start():
    global _llm
    _ensure_vs_ready()
    _llm = LLM(settings)
    # Apply shim if available to maintain Phi-3/context behavior
    _llm = apply_context_shim(_llm, settings)
    _llm.warmup()

# ----------------------------
# Health/Admin
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "vectorstore_ok": bool(_VECTORSTORE_OK), "emb_backend": _EMB_BACKEND_MOUNTED, "context_shim": _CONTEXT_SHIM_OK}

@app.get("/admin/info")
def admin_info():
    try:
        st = _vs.status() if _vs else {}
    except Exception as e:
        st = {"exists": False, "count": 0, "size_bytes": 0, "index_dir": str(settings.chroma_db_dir), "error": str(e)}

    def _safe_gpu_info():
        try:
            return gpu_info()
        except Exception as e:
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}

    llama = {"cuda": _llm.is_cuda() if _llm else False, "ctx": _llm.ctx_size() if _llm else 0}
    info = {
        "ok": True,
        "docs_dir": str(settings.documents_dir),
        "index_dir": st.get("index_dir", str(settings.chroma_db_dir)),
        "index_exists": st.get("exists", False),
        "doc_count": st.get("count", 0),
        "index_size_bytes": st.get("size_bytes", 0),
        "model_path": str(settings.model_path),
        "embed_backend": _EMB_BACKEND_MOUNTED,
        "gpu": _safe_gpu_info(),
        "llama": llama,
        "vectorstore_ok": bool(_VECTORSTORE_OK),
        "vectorstore_error": (None if _VECTORSTORE_OK else "VectorStore disabled (missing dependency, e.g., PyMuPDF/fitz)"),
        "context_shim": _CONTEXT_SHIM_OK,
    }
    if hasattr(app.state, "_emb_import_error"):
        info["emb_import_error"] = str(app.state._emb_import_error)
    return info

@app.get("/admin/index_stats")
def admin_index_stats():
    try:
        idx_dir = chroma_dir_global(settings.chroma_db_dir)
    except Exception:
        idx_dir = settings.chroma_db_dir
    size = 0
    try:
        p = Path(idx_dir)
        size = int(db_size_bytes(idx_dir) if p.exists() else 0)
    except Exception:
        size = 0
    return {
        "index_dir": str(idx_dir),
        "index_size_bytes": size,
        "docs_dir": str(settings.documents_dir),
        "docs_dir_exists": Path(settings.documents_dir).exists(),
        "vectorstore_ok": bool(_VECTORSTORE_OK),
    }

@app.get("/admin/version")
def admin_version():
    ver = None
    try:
        if _pkg_version is not None:
            ver = _pkg_version("ed_core")
    except Exception:
        ver = None
    return {"package": "ed_core", "version": ver}

@app.get("/admin/gpu_check")
def admin_gpu_check():
    info = {"torch": None, "onnxruntime": None, "tiny_matmul": None}
    try:
        import torch as _t
        info["torch"] = {
            "available": bool(_t.cuda.is_available()),
            "device_count": int(_t.cuda.device_count() if _t.cuda.is_available() else 0),
            "name": _t.cuda.get_device_name(0) if _t.cuda.is_available() else None,
        }
        if _t.cuda.is_available():
            a = _t.randn(64, 64, device="cuda"); b = _t.randn(64, 64, device="cuda")
            info["tiny_matmul"] = float((a @ b).sum().item())
    except Exception as e:
        info["torch"] = {"error": str(e)}

    try:
        import onnxruntime as _ort
        info["onnxruntime"] = {
            "get_device": _ort.get_device() if hasattr(_ort, "get_device") else "unknown",
            "providers": _ort.get_available_providers() if hasattr(_ort, "get_available_providers") else [],
        }
    except Exception as e:
        info["onnxruntime"] = {"error": str(e)}

    info["api_llm_cuda"] = bool(_llm.is_cuda()) if _llm else False
    return info

@app.get("/admin/persona")
def admin_persona():
    return {"persona_head": PERSONA[:400], "length": len(PERSONA)}

@app.post("/admin/echo_prompt")
def admin_echo_prompt(payload=Body(...)):
    q = _normalize_question(str(
        payload.get("query", "") or payload.get("q", "") or payload.get("question", "")
    ).strip())
    if not q:
        return {"system": "", "user": "", "ctx_count": 0}
    k = max(6, getattr(settings, "rag_top_k", 4))
    thr = min(0.05, getattr(settings, "rag_threshold", 0.10))
    procedure_mode = _is_procedure_query(q)

    t_r0 = time.time()
    hits = _collect_hits_with_rerank(q, k=k, thr=thr, procedure_mode=procedure_mode) if _VECTORSTORE_OK else []
    retrieve_s = round(time.time() - t_r0, 3)

    if hits:
        if procedure_mode:
            prof = _domain_profile(q)
            max_steps = 8 if prof["name"] == "loto" else 4
            ctx_lines, _srcs, _ = _build_procedure_context(hits, total_ctx_chars=1200, window_pages=2, max_steps=max_steps, q=q)
            sys_prompt = (
                PERSONA + "\n" + CONFIDENCE_POLICY +
                "You are ED. Synthesize a clear, numbered procedure based ONLY on the context. "
                "Write actionable steps; do not copy publication lists or TOC entries. "
                "Each step MUST include a citation like [filename.pdf p.X]. Keep it concise.\n"
            )
        else:
            ctx_lines, _srcs, _ = _build_nonproc_context(hits, max_ctx_chars=1200, window_pages=0, q=q)
            sys_prompt = (
                PERSONA + "\n" + CONFIDENCE_POLICY +
                "You are ED. Formulate a 2–4 paragraph, shop-floor explanation synthesized from the context. "
                "Do not output publication/resource lists; integrate facts into the prose. "
                "Include at least one inline citation per paragraph like [filename.pdf p.X].\n"
            )
        user_msg = f"Question: {q}\n\nContext:\n" + "\n".join(ctx_lines) + "\n\nAnswer:"
    else:
        sys_prompt = PERSONA + "\n" + CONFIDENCE_POLICY + "No useful context found. Use general knowledge if appropriate."
        user_msg = f"Question: {q}\n\nAnswer:"
    return {"system": sys_prompt[:1000], "user": user_msg[:1000], "ctx_count": len(hits), "retrieve_s": retrieve_s}

@app.post("/admin/preview")
def admin_preview(payload=Body(...)):
    q = _normalize_question(str(
        payload.get("query", "") or payload.get("q", "") or payload.get("question", "")
    ).strip())
    if not q:
        return {"ok": False, "reason": "empty"}
    mode = "rag_general"
    if _is_identity_query(q): mode = "identity"
    elif _is_offdomain_query(q): mode = "offdomain"
    elif not _is_in_scope_query(q): mode = "pivot"
    proc = _is_procedure_query(q)
    k = max(6, getattr(settings, "rag_top_k", 4))
    thr = min(0.05, getattr(settings, "rag_threshold", 0.10))

    t_r0 = time.time()
    hits = _collect_hits_with_rerank(q, k=k, thr=thr, procedure_mode=proc) if (mode.startswith("rag") and _VECTORSTORE_OK) else []
    retrieve_s = round(time.time() - t_r0, 3)

    if hits:
        prof = _domain_profile(q)
        max_steps = 8 if (proc and prof["name"] == "loto") else 4
        ctx_lines, sources, citations = (
            _build_procedure_context(hits, 1200, 2, max_steps=max_steps, q=q) if proc else _build_nonproc_context(hits, 1200, 0, q=q)
        )
    else:
        ctx_lines, sources, citations = ([], [], [])
    return {
        "ok": True,
        "mode": "rag_procedure" if (mode.startswith("rag") and proc) else mode,
        "ctx_count": len(ctx_lines),
        "sources": sources[:8],
        "citations": citations[:8],
        "sample_context": ctx_lines[:6],
        "commands_detected": _detect_ed_cmds(q),
        "vectorstore_ok": bool(_VECTORSTORE_OK),
        "retrieve_s": retrieve_s
    }

@app.post("/admin/rebuild")
def admin_rebuild():
    drop_db(settings.chroma_db_dir)
    _ensure_vs_ready()
    return {"status": "ok", "vectorstore_ok": bool(_VECTORSTORE_OK)}

# ----------------------------
# Main /query
# ----------------------------
@app.post("/query")
def query(payload=Body(...)):
    q_raw = str(payload.get("query", "") or payload.get("q", "") or payload.get("question", "")).strip()
    session_id = str(payload.get("session_id", ""))
    q = _normalize_question(q_raw)

    force_summary = bool(payload.get("summary", False))
    max_summary_sentences = int(payload.get("max_sentences", 4))
    if _wants_two_sentence_cites(q):
        force_summary = True
        max_summary_sentences = 2

    if not q:
        return {
            "answer": "Please ask a question.",
            "sources": [],
            "citations": [],
            "allow_open": getattr(settings, "allow_open", False),
            "timing": {"total_s": 0.0, "retrieve_s": 0.0, "generate_s": 0.0}
        }

    t0 = time.time()
    t0_iso = datetime.utcnow().isoformat()+"Z"

    # 0) Commands
    cmds = _detect_ed_cmds(q)
    ed_cmd = cmds[0] if cmds else None
    if cmds and _is_pure_command(q, cmds):
        elapsed = round(time.time() - t0, 3)
        t1_iso = datetime.utcnow().isoformat()+"Z"
        answer_json = json.dumps(ed_cmd, separators=(",", ":"), ensure_ascii=False)
        resp = {
            "answer": answer_json,
            "sources": [],
            "citations": [],
            "allow_open": getattr(settings, "allow_open", False),
            "ed_cmd": ed_cmd,
            "ed_cmds": cmds,
            "mode": "command",
            "topic_area": "general",
            "ctx_count": 0,
            "timing": {
                "total_s": elapsed,
                "total_ms": int(elapsed*1000),
                "retrieve_s": 0.0,
                "generate_s": 0.0,
                "start_utc": t0_iso,
                "end_utc": t1_iso,
                "ansi": _ansi(f"{elapsed:.3f}s", "cyan")
            },
            "vectorstore_ok": bool(_VECTORSTORE_OK),
            "ansi_sources": [],
        }
        _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": "command", "q": q, "elapsed_s": elapsed, "ctx_count": 0, "cmds": cmds})
        return resp

    # 1) Identity / origin (TIMED)
    if _is_identity_query(q):
        ans, gen_s = _dynamic_identity_answer(session_id)
        elapsed = round(time.time() - t0, 3)
        t1_iso = datetime.utcnow().isoformat()+"Z"
        ans = _force_ed_voice(_sanitize_identity_leaks(ans, fallback_identity=True))
        resp = {
            "answer": ans,
            "sources": [],
            "citations": [],
            "allow_open": getattr(settings, "allow_open", False),
            "ed_cmd": ed_cmd,
            "ed_cmds": cmds or [],
            "mode": "identity",
            "topic_area": "general",
            "ctx_count": 0,
            "timing": {
                "total_s": elapsed,
                "total_ms": int(elapsed*1000),
                "retrieve_s": 0.0,
                "generate_s": gen_s,
                "start_utc": t0_iso,
                "end_utc": t1_iso,
                "ansi": _ansi(f"{elapsed:.3f}s","green")
            },
            "vectorstore_ok": bool(_VECTORSTORE_OK),
            "ansi_sources": [],
        }
        _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": "identity", "q": q, "elapsed_s": elapsed, "ctx_count": 0, "gen_s": gen_s})
        return resp

    # 2) Off-domain pivot (TIMED)
    if _is_offdomain_query(q):
        topic = _short_topic_phrase(q)
        prof = _domain_profile(q)
        fups = "; ".join(_suggest_followups(prof["name"])[:2])
        system = PERSONA + "\nUse first person as ED. Keep it brief (1–2 sentences). Be specific about the topic."
        user = (
            f"You are ED. The user asked about: {topic}. "
            "Politely explain your offline scope (no live finance/news or emotions). "
            "Pivot to shop-floor topics and suggest one concrete follow-up in the same sentence if possible."
        )
        out, gen_s = _llm_generate_timed(system, user, max_tokens=min(90, _max_tokens_for_budget(t0, 90)), temp=0.5)
        out = _clip_trailing_fragments(_force_ed_voice(_sanitize_identity_leaks(out)))
        out = _truncate_to_two_sentences(out)
        bad_tail = out.rstrip().endswith("[")
        junk_markers = ("shopping", "deals", "trending", "subscribe", "follow me")
        too_long = len(out.split()) > 60
        if (hasattr(_llm, "looks_bad") and _llm.looks_bad(out)) or bad_tail or any(j in out.lower() for j in junk_markers) or too_long:
            out = _truncate_to_two_sentences(
                f"You asked about {topic}. I run fully offline on a Jetson—no live finance/news or feelings. "
                f"I can help with robotics, safety, Lean, and ISO/IATF. Try: {fups}."
            )
        elapsed = round(time.time() - t0, 3)
        t1_iso = datetime.utcnow().isoformat()+"Z"
        resp = {
            "answer": out,
            "sources": [],
            "citations": [],
            "allow_open": getattr(settings, "allow_open", False),
            "ed_cmd": ed_cmd,
            "ed_cmds": cmds or [],
            "mode": "offdomain",
            "topic_area": prof["name"],
            "ctx_count": 0,
            "timing": {
                "total_s": elapsed,
                "total_ms": int(elapsed*1000),
                "retrieve_s": 0.0,
                "generate_s": gen_s,
                "start_utc": t0_iso,
                "end_utc": t1_iso,
                "ansi": _ansi(f"{elapsed:.3f}s","yellow")
            },
            "vectorstore_ok": bool(_VECTORSTORE_OK),
            "ansi_sources": [],
        }
        _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": "offdomain", "q": q, "elapsed_s": elapsed, "ctx_count": 0, "gen_s": gen_s})
        return resp

    # 3) Out-of-scope pivot (TIMED)
    if not _is_in_scope_query(q):
        topic = _short_topic_phrase(q)
        prof = _domain_profile(q)
        fups = "; ".join(_suggest_followups(prof["name"])[:2])
        system = PERSONA + "\nSpeak as ED in first person. Short, confident, shop-floor tone (≤2 sentences)."
        user = (
            f"You are ED. The user asked: {topic}. "
            "Briefly acknowledge the topic, explain your remit (offline, factory-floor), "
            "and suggest 1 concrete follow-up in-scope."
        )
        out, gen_s = _llm_generate_timed(system, user, max_tokens=min(90, _max_tokens_for_budget(t0, 90)), temp=0.5)
        out = _clip_trailing_fragments(_force_ed_voice(_sanitize_identity_leaks(out)))
        out = _truncate_to_two_sentences(out)
        if hasattr(_llm, "looks_bad") and _llm.looks_bad(out):
            out = _truncate_to_two_sentences(
                f"I may not cover {topic} directly. I’m ED—offline on the shop floor. Try: {fups}."
            )
        elapsed = round(time.time() - t0, 3)
        t1_iso = datetime.utcnow().isoformat()+"Z"
        resp = {
            "answer": out,
            "sources": [],
            "citations": [],
            "allow_open": getattr(settings, "allow_open", False),
            "ed_cmd": ed_cmd,
            "ed_cmds": cmds or [],
            "mode": "pivot",
            "topic_area": prof["name"],
            "ctx_count": 0,
            "timing": {
                "total_s": elapsed,
                "total_ms": int(elapsed*1000),
                "retrieve_s": 0.0,
                "generate_s": gen_s,
                "start_utc": t0_iso,
                "end_utc": t1_iso,
                "ansi": _ansi(f"{elapsed:.3f}s","magenta")
            },
            "vectorstore_ok": bool(_VECTORSTORE_OK),
            "ansi_sources": [],
        }
        _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": "pivot", "q": q, "elapsed_s": elapsed, "ctx_count": 0, "gen_s": gen_s})
        return resp

    # 4) RAG path (TIMED: separate retrieve vs generate)
    k = max(6, getattr(settings, "rag_top_k", 4))
    thr = min(0.05, getattr(settings, "rag_threshold", 0.10))
    procedure_mode = _is_procedure_query(q)
    if force_summary:
        procedure_mode = False

    t_r0 = time.time()
    hits = _collect_hits_with_rerank(q, k=k, thr=thr, procedure_mode=procedure_mode) if _VECTORSTORE_OK else []
    retrieve_s = round(time.time() - t_r0, 3)

    # RULEBOOK fallback
    if not hits and _ED_RULEBOOK_FALLBACK:
        rb = _rulebook_match(q)
        if rb:
            out = _force_ed_voice(_sanitize_identity_leaks(rb["answer"]))
            elapsed = round(time.time() - t0, 3)
            t1_iso = datetime.utcnow().isoformat()+"Z"
            prof = _domain_profile(q)
            resp = {
                "answer": out,
                "sources": [],
                "citations": [],
                "allow_open": getattr(settings, "allow_open", False),
                "ed_cmd": None,
                "ed_cmds": [],
                "mode": "rulebook",
                "topic_area": prof["name"],
                "ctx_count": 0,
                "timing": {
                    "total_s": elapsed,
                    "total_ms": int(elapsed*1000),
                    "retrieve_s": retrieve_s,
                    "generate_s": 0.0,
                    "start_utc": t0_iso,
                    "end_utc": t1_iso,
                    "ansi": _ansi(f"{elapsed:.3f}s","blue")
                },
                "vectorstore_ok": bool(_VECTORSTORE_OK),
                "ansi_sources": [],
            }
            _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": "rulebook", "rule": rb.get("rule"), "q": q,
                        "elapsed_s": elapsed, "ctx_count": 0, "retrieve_s": retrieve_s})
            return resp

    ctx_lines: list[str] = []; srcs: list[str] = []; citations: list[dict] = []
    if hits:
        if procedure_mode:
            prof = _domain_profile(q)
            max_steps = 8 if prof["name"] == "loto" else 4
            ctx_lines, srcs, citations = _build_procedure_context(hits, total_ctx_chars=1200, window_pages=2, max_steps=max_steps, q=q)
        else:
            ctx_lines, srcs, citations = _build_nonproc_context(hits, max_ctx_chars=1200, window_pages=0, q=q)
    else:
        srcs = []; citations = []

    # LLM answer (TIMED)
    answer = ""; gen_s = 0.0
    if _ED_USE_LLM_SUMMARY:
        try:
            answer, gen_s = _llm_cited_answer(
                q, ctx_lines, two_sentence=_wants_two_sentence_cites(q) or force_summary,
                procedure_mode=procedure_mode, t0=t0,
            )
        except Exception as e:
            answer = f"(internal generation error: {e})"
            gen_s = 0.0
        if procedure_mode:
            if (hasattr(_llm, "looks_bad") and _llm.looks_bad(answer)) or not re.search(r"^\d+[\.\)]\s", answer, re.M) or _looks_like_reference_list(answer):
                answer = ""
        else:
            if (hasattr(_llm, "looks_bad") and _llm.looks_bad(answer)) or looks_like_ref_dump(answer):
                answer = ""

    if not answer:
        if procedure_mode:
            answer = ensure_numbered_with_cites("", ctx_lines)
        else:
            target_sents = 2 if _wants_two_sentence_cites(q) else (max_summary_sentences if force_summary else 4)
            synthesized = _compose_answer_from_ctx(q, ctx_lines, max_sentences=target_sents) if ctx_lines else ""
            if synthesized and (not hasattr(_llm, "looks_bad") or not _llm.looks_bad(synthesized)) and not looks_like_ref_dump(synthesized):
                answer = synthesized
            else:
                if ctx_lines and _ED_STRICT_RAG:
                    answer = cited_sentences_from_ctx(ctx_lines, max_sentences=target_sents)
                    if _wants_two_sentence_cites(q):
                        answer = _enforce_two_sentence_cites(answer, ctx_lines)
                else:
                    answer = answer or "I don’t have enough grounded context to answer that."

    if looks_like_ref_dump(answer) and ctx_lines:
        answer = _compose_answer_from_ctx(q, ctx_lines, max_sentences=2) or answer

    if (not answer) or answer.strip().lower().startswith("i don’t have enough"):
        t = _templated_short_answer(q)
        if t:
            answer = _force_ed_voice(_sanitize_identity_leaks(t))

    used_srcs, used_cites = _citations_from_answer(answer)
    if used_cites:
        srcs = used_srcs; citations = used_cites

    elapsed = round(time.time() - t0, 3)
    t1_iso = datetime.utcnow().isoformat()+"Z"
    mode = "rag_procedure" if procedure_mode else ("no_context" if not hits else "rag_general")
    resp = {
        "answer": answer,
        "sources": srcs,
        "citations": citations,
        "allow_open": getattr(settings, "allow_open", False),
        "ed_cmd": ed_cmd,
        "ed_cmds": cmds or [],
        "mode": mode,
        "topic_area": _domain_profile(q)["name"],
        "ctx_count": len(ctx_lines),
        "timing": {
            "total_s": elapsed,
            "total_ms": int(elapsed*1000),
            "retrieve_s": retrieve_s,
            "generate_s": gen_s,
            "start_utc": t0_iso,
            "end_utc": t1_iso,
            "ansi": _ansi(f"{elapsed:.3f}s","cyan")
        },
        "vectorstore_ok": bool(_VECTORSTORE_OK),
        "ansi_sources": _ansi_sources_list(srcs),
    }

    _log_query({"ts": datetime.utcnow().isoformat()+"Z", "mode": mode, "q": q, "elapsed_s": elapsed,
                "ctx_count": len(ctx_lines), "sources": srcs[:6], "retrieve_s": retrieve_s, "gen_s": gen_s})

    return resp

def main():
    import uvicorn
    uvicorn.run("ed_core.rag_service.api:app", host="0.0.0.0", port=8000, reload=False)
