# ed_core/rag_service/services/llm.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import re, inspect, time, os

try:
    from llama_cpp import Llama as _Llama
except Exception:
    _Llama = None

from ..config import settings


class LLM:
    """Thin wrapper around llama.cpp with a deterministic stub fallback."""
    _FALLBACK_MSG = (
        "I’m not confident in the answer I produced. Please rephrase or specify the machine/model so I can ground the reply."
    )

    def __init__(self, cfg=settings):
        self._cfg = cfg

        # --- decoding / runtime knobs (with safe defaults if missing) ---
        self._ctx = int(getattr(cfg, "ctx_size", 4096))
        self._seed = int(getattr(cfg, "seed", 0))
        self._n_batch = int(getattr(cfg, "n_batch", 128))
        self._n_gpu_layers = int(getattr(cfg, "n_gpu_layers", -1))  # -1 = auto, 0 = CPU
        self._use_mlock = bool(int(getattr(cfg, "use_mlock", 0)))
        self._llm_retries = int(getattr(cfg, "llm_retries", 2))

        # Phi-3 stop token; also accept common alternatives if a different GGUF is used
        self._phi3_eos = "<|end|>"
        self._stop_default: List[str] = [self._phi3_eos, "<|eot_id|>", "</s>"]

        self._llm: Optional[Any] = None
        self._allowed_chat: set[str] = set()
        self._allowed_comp: set[str] = set()

        # Heuristic "CUDA?" (llama.cpp doesn't expose a direct flag)
        self._is_cuda_flag = (self._n_gpu_layers != 0)

        # --- try to init llama.cpp if present & model exists ---
        if _Llama is not None:
            try:
                model_path = str(cfg.model_path)
                if os.path.exists(model_path):
                    self._llm = _Llama(
                        model_path=model_path,
                        n_ctx=self._ctx,
                        n_batch=min(256, self._n_batch),
                        n_gpu_layers=self._n_gpu_layers,
                        seed=self._seed,
                        verbose=False,
                        add_bos_token=False,
                        use_mmap=True,
                        use_mlock=self._use_mlock,
                    )
                    # discover supported kwargs for robust filtering
                    self._allowed_chat = set(inspect.signature(self._llm.create_chat_completion).parameters.keys())
                    try:
                        self._allowed_comp = set(inspect.signature(self._llm.create_completion).parameters.keys())
                    except Exception:
                        self._allowed_comp = set()
                else:
                    # model path missing → stay on stub
                    self._llm = None
            except Exception:
                self._llm = None
                self._allowed_chat = set()
                self._allowed_comp = set()

        # Capability flags (llama.cpp builds differ)
        self._has_repeat_last_n_chat = ("repeat_last_n" in self._allowed_chat)
        self._has_penalty_last_n_chat = ("penalty_last_n" in self._allowed_chat)
        self._has_repeat_last_n_comp = ("repeat_last_n" in self._allowed_comp)
        self._has_penalty_last_n_comp = ("penalty_last_n" in self._allowed_comp)

    # ---- health ----
    def is_cuda(self) -> bool:
        return bool(self._llm is not None and self._is_cuda_flag)

    def ctx_size(self) -> int:
        return int(self._ctx)

    def release(self):
        self._llm = None
        self._allowed_chat.clear()
        self._allowed_comp.clear()

    def warmup(self):
        """Fast one-token run to prime the context cache; safe no-op if stub."""
        if not self._llm:
            return
        try:
            self._llm.create_chat_completion(
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                temperature=0.1,
                stop=self._stop_default,
            )
        except Exception:
            pass

    # ---- output guards ----
    @staticmethod
    def looks_bad(text: str) -> bool:
        if not text:
            return True
        s = text.strip()
        if len(s) < 6:
            return True
        # extremely long repeated char runs
        if re.search(r"(.)\1{20,}", s):
            return True
        # signal soup after removing bracket cites and list numerals
        clean = re.sub(r"\[[^\]]+\]", "", s)
        clean = re.sub(r"^\s*[-•\d]+\.\s*", "", clean, flags=re.MULTILINE)
        letters = len(re.findall(r"[A-Za-z0-9]", clean))
        nonword = len(re.findall(r"[^A-Za-z0-9\s]", clean))
        if letters > 0 and (nonword / max(1, letters)) > 3.0:
            return True
        return False

    @classmethod
    def is_fallback(cls, text: str) -> bool:
        return bool(text and text.strip().startswith(cls._FALLBACK_MSG[:24]))

    # ---- llama helpers ----
    def _filter_chat_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trim kwargs to what the current llama.cpp build supports and merge stop tokens."""
        if not self._llm:
            return {}
        p = dict(params)

        # Handle repeat_last_n/penalty_last_n naming differences
        if "repeat_last_n" in p and "repeat_last_n" not in self._allowed_chat:
            if "penalty_last_n" in self._allowed_chat:
                p["penalty_last_n"] = p.pop("repeat_last_n")
            else:
                p.pop("repeat_last_n", None)

        # Merge caller-provided stop with our defaults (dedup, stable order)
        stop_from_caller = p.pop("stop", None)
        if stop_from_caller:
            merged: List[str] = []
            for tok in list(stop_from_caller) + self._stop_default:
                if tok and tok not in merged:
                    merged.append(tok)
            p["stop"] = merged
        else:
            p["stop"] = list(self._stop_default)

        # Only pass supported params
        return {k: v for k, v in p.items() if k in self._allowed_chat}

    def _chat_llama_once(self, system: str, user: str, **kw) -> str:
        params = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        params.update(self._filter_chat_kwargs(kw))
        resp: Dict[str, Any] = self._llm.create_chat_completion(**params)
        return resp["choices"][0]["message"]["content"]

    def _chat_llama(self, system: str, user: str, **kw) -> str:
        if not self._llm:
            return ""
        attempts = max(0, self._llm_retries)
        base_delay = 0.08
        last_err = None
        for i in range(attempts + 1):
            try:
                return self._chat_llama_once(system, user, **kw)
            except Exception as e:
                last_err = e
                time.sleep(base_delay * (1.6 ** i))
        # One more conservative pass with safer defaults
        try:
            return self._chat_llama_once(
                system, user,
                max_tokens=kw.get("max_tokens", 128),
                temperature=max(0.2, min(kw.get("temperature", 0.35), 0.6)),
                top_p=min(0.9, kw.get("top_p", 0.9)),
                top_k=min(40, kw.get("top_k", 60)),
                typical_p=kw.get("typical_p", 0.95),
                min_p=kw.get("min_p", 0.05),
                repeat_penalty=max(1.05, kw.get("repeat_penalty", 1.12)),
                repeat_last_n=min(320, kw.get("repeat_last_n", 256)),
                frequency_penalty=kw.get("frequency_penalty", 0.0),
                presence_penalty=kw.get("presence_penalty", 0.0),
                mirostat_mode=0,
                stop=kw.get("stop", None),
            ).strip()
        except Exception:
            return ""

    # ---- stub path (when llama.cpp is unavailable) ----
    @staticmethod
    def _parse_context(user: str) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
        q_match = re.search(r"Question:\s*(.+?)\n\nContext:", user, flags=re.S)
        question = q_match.group(1).strip() if q_match else ""
        m = re.search(r"Context:\s*(.+?)\n\nAnswer:", user, flags=re.S)
        if not m:
            return question, []
        raw = m.group(1)
        items: List[Tuple[str, Optional[str]]] = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            m2 = re.match(r"^\-\s*\[([^\]]+)\]\s*(.+)$", ln)
            if m2:
                items.append((m2.group(2).strip(), m2.group(1).strip()))
                continue
            m3 = re.match(r"^\-\s*(.+)$", ln)
            if m3:
                items.append((m3.group(1).strip(), None))
        return question, items

    @staticmethod
    def _is_procedural(system: str, user: str, question: str) -> bool:
        hay = " ".join([system.lower(), user.lower(), question.lower()])
        keys = ["extract an ordered, numbered list of steps", "steps", "procedure", "checklist", "startup", "shutdown", "calibrate"]
        return any(k in hay for k in keys)

    def _stub_generate(self, system: str, user: str, max_tokens: int) -> str:
        question, ctx_items = self._parse_context(user)
        if not ctx_items:
            return self._FALLBACK_MSG
        procedural = self._is_procedural(system, user, question)
        limit = max(3, min(8, max_tokens // 40))
        if procedural:
            out = []
            for i, (snippet, cite) in enumerate(ctx_items[:limit], 1):
                sent = re.split(r'(?<=[.!?])\s+', snippet)[0].strip()
                if len(sent) > 220:
                    sent = sent[:220].rstrip() + "…"
                out.append(f"{i}. {sent}" + (f" [{cite}]" if cite else ""))
            if not any(ch.isdigit() for (snip, _) in ctx_items for ch in snip[:6]):
                out.append("Limitations: Steps are inferred from nearby context; confirm machine-specific details.")
            return "\n".join(out)
        bullets = []
        for (snippet, cite) in ctx_items[:limit]:
            sn = snippet.strip()
            if len(sn) > 240:
                sn = sn[:240].rstrip() + "…"
            bullets.append(("• " + sn) + (f" [{cite}]" if cite else ""))
        return "From local docs:\n" + "\n".join(bullets)

    # ---- public API ----
    def generate(self, system: str, user: str, max_tokens: int = 192, temp: float = 0.4, **decode):
        """Chat-complete with llama.cpp, with a deterministic local fallback."""
        if self._llm is not None:
            try:
                text = self._chat_llama(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=max(0.2, min(temp, 0.6)),
                    top_p=decode.get("top_p", 0.92),
                    top_k=decode.get("top_k", 60),
                    typical_p=decode.get("typical_p", 0.95),
                    min_p=decode.get("min_p", 0.06),
                    repeat_penalty=decode.get("repeat_penalty", 1.12),
                    repeat_last_n=decode.get("repeat_last_n", 256),
                    frequency_penalty=decode.get("frequency_penalty", 0.0),
                    presence_penalty=decode.get("presence_penalty", 0.0),
                    mirostat_mode=decode.get("mirostat_mode", 0),
                    seed=decode.get("seed", self._seed),
                    grammar=decode.get("grammar", None),
                    stop=decode.get("stop", None),
                ).strip()
                if text and not self.looks_bad(text):
                    return text

                # Second try: slightly different decode mix; ask to be concise
                text2 = self._chat_llama(
                    system=system,
                    user=user + "\n\nBe concise. Avoid repeated symbols.",
                    max_tokens=max_tokens,
                    temperature=0.35,
                    top_p=0.9,
                    top_k=50,
                    typical_p=0.95,
                    min_p=0.05,
                    repeat_penalty=1.15,
                    repeat_last_n=320,
                    frequency_penalty=0.1,
                    presence_penalty=0.05,
                    mirostat_mode=0,
                    stop=decode.get("stop", None),
                ).strip()
                if text2 and not self.looks_bad(text2):
                    return text2
            except Exception:
                # fall through to stub
                pass

        stub = self._stub_generate(system, user, max_tokens=max_tokens)
        return stub if stub else self._FALLBACK_MSG
