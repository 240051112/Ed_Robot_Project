"""
Reusable summarization helpers for ED's RAG answers.
These mirror the guardrail helpers in api.py so other services can
produce the same style of output without duplicating logic.
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import re

PDF_CITE_RE = re.compile(r"\[[^\]]+\.pdf(?:\s+p\.\d+)?\]", re.IGNORECASE)

def cited_sentences_from_ctx(ctx_lines: List[str], max_sentences: int = 4) -> str:
    out: List[str] = []
    for line in ctx_lines[: max_sentences + 1]:
        m = re.match(r"^- \[([^\]]+)\]\s*(.*)$", line)
        cite = m.group(1) if m else ""
        body = (m.group(2) if m else line.lstrip("- ").strip())
        sent = re.split(r'(?<=[.!?])\s+', body)[0]
        sent = " ".join(sent.split())
        if len(sent) > 220:
            sent = sent[:220].rsplit(" ", 1)[0] + "…"
        out.append(sent + (f" [{cite}]" if cite else ""))
        if len(out) >= max_sentences:
            break
    return " ".join(out) if out else ""

def looks_like_ref_dump(text: str) -> bool:
    if not text:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ref_hits = 0
    for l in lines[:6]:
        if re.search(r"(OSHA|ANSI|ISO|IATF|Manual|Publication|eTool)", l, re.I):
            ref_hits += 1
        if re.search(r"\b(p\.\d+)\b", l):
            ref_hits += 1
        if re.match(r"^\d+[\.\)]\s", l):
            ref_hits += 1
    if ref_hits >= 3:
        return True
    cites = PDF_CITE_RE.findall(text)
    if len(cites) >= 2:
        if not lines:
            return True
        long_line = max(lines, key=len)
        return (len(long_line) > 120) or (len(lines) <= 3)
    return False

def ensure_numbered_with_cites(raw: str, ctx_lines: List[str]) -> str:
    cites: List[str] = []
    for line in ctx_lines:
        m = re.match(r"^- \[([^\]]+)\]", line)
        if m:
            cites.append(m.group(1))

    if not raw or looks_like_ref_dump(raw):
        steps: List[str] = []
        for i, line in enumerate(ctx_lines[:8], 1):
            m = re.match(r"^- \[([^\]]+)\]\s*(.*)$", line)
            cite = m.group(1) if m else (cites[i-1] if i-1 < len(cites) else "")
            txt = (m.group(2) if m else line.lstrip("- ").strip())
            first = re.split(r'(?<=[.!?])\s+', txt.strip())[0].rstrip(".")
            steps.append(f"{i}. {first}" + (f" [{cite}]" if cite else ""))
        return "\n".join(steps) if steps else raw

    out_lines: List[str] = []
    num = 1
    for raw_line in raw.splitlines():
        l = raw_line.strip()
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
        has_cite = bool(PDF_CITE_RE.search(step_txt))
        cite = cites[(num-1) % len(cites)] if cites else ""
        out_lines.append(f"{num}. {step_txt}" + ("" if has_cite or not cite else f" [{cite}]"))
        num += 1
    return "\n".join(out_lines[:10]) if out_lines else raw
