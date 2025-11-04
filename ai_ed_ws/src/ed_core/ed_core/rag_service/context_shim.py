from pathlib import Path
from typing import Any, Dict, List, Tuple

def _basename_only(path_like: str | None) -> str:
    if not path_like:
        return "unknown"
    try:
        return Path(path_like).name
    except Exception:
        return str(path_like).split("/")[-1]

def _pull_text_and_meta(h: Any) -> tuple[str, dict]:
    # Works with dict hits or simple objects
    text = (
        getattr(h, "page_content", None)
        or getattr(h, "content", None)
        or (h.get("page_content") if isinstance(h, dict) else None)
        or (h.get("content") if isinstance(h, dict) else None)
        or (h.get("text") if isinstance(h, dict) else None)
        or ""
    )
    meta = (
        getattr(h, "metadata", None)
        or (h.get("metadata") if isinstance(h, dict) else None)
        or {}
    )
    # fallbacks
    if isinstance(h, dict):
        if "source" not in meta and "source" in h:
            meta["source"] = h["source"]
        if "page" not in meta and "page" in h:
            meta["page"] = h["page"]
    return str(text or ""), dict(meta or {})

def _mk_label(meta: Dict[str, Any]) -> tuple[str, str, int | None]:
    src = meta.get("title") or meta.get("source") or meta.get("path") or "unknown"
    page = meta.get("page") or meta.get("page_number") or meta.get("pageIndex")
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None
    base = _basename_only(src)
    label = f"{base}{f' p.{page}' if page is not None else ''}"
    return label, base, page

def _clean(s: str, max_len: int = 240) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= max_len else (s[: max_len - 1].rstrip() + "…")

def build_nonproc_context(
    hits: List[Dict[str, Any]],
    max_ctx_chars: int = 1200,
    window_pages: int = 0,  # ignored in shim
    q: str | None = None,
) -> Tuple[List[str], List[str], List[Dict[str, int]]]:
    """
    Return:
      - ctx_lines: '- [file.pdf p.X] snippet' per line
      - sources:   unique labels used
      - citations: dicts {source: 'file.pdf', page: int|None}
    """
    lines: List[str] = []
    sources: List[str] = []
    cites: List[Dict[str, int]] = []
    total = 0

    for h in hits or []:
        text, meta = _pull_text_and_meta(h)
        if not text.strip():
            continue
        label, base, page = _mk_label(meta)
        snip = _clean(text, 280)
        if total + len(snip) > max_ctx_chars:
            snip = snip[: max(0, max_ctx_chars - total)]
        if not snip:
            break
        lines.append(f"- [{label}] {snip}")
        total += len(snip)
        if label not in sources:
            sources.append(label)
        cites.append({"source": base, "page": page if page is not None else 0})
        if total >= max_ctx_chars:
            break

    return lines, sources, cites

def build_procedure_context(
    hits: List[Dict[str, Any]],
    total_ctx_chars: int = 1200,
    window_pages: int = 1,  # ignored in shim
    max_steps: int = 6,
    q: str | None = None,
) -> Tuple[List[str], List[str], List[Dict[str, int]]]:
    """
    Simple fallback: reuse non-proc lines and just truncate to ~max_steps items worth of text.
    """
    lines, sources, cites = build_nonproc_context(hits, max_ctx_chars=total_ctx_chars, q=q)
    return lines[: max_steps * 3], sources, cites
