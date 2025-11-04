# boot.py — start llama.cpp server (Phi-3 on CUDA) then ED FastAPI
from __future__ import annotations
import os, subprocess, sys, time, socket, signal
from pathlib import Path

LLAMA_BIN_DEFAULT = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"

def wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except Exception:
            time.sleep(0.25)
    return False

def start_llama_server():
    model = os.getenv("ED_MODEL_PATH") or os.getenv("MODEL") or str(Path.home()/"phi3_models"/"Phi-3-mini-4k-instruct.Q4_K_M.gguf")
    host  = os.getenv("ED_LLM_HOST", "127.0.0.1")
    port  = int(os.getenv("ED_LLM_PORT", "8080"))
    ngl   = os.getenv("ED_LLM_NGL", "999")
    ctx   = os.getenv("ED_LLM_CTX", "4096")
    batch = os.getenv("ED_LLM_BATCH", "256")
    temp  = os.getenv("ED_LLM_TEMP", "0.4")
    bin_path = os.getenv("ED_LLAMA_SERVER_BIN", str(LLAMA_BIN_DEFAULT))

    if not Path(model).exists():
        print(f"[boot] ERROR: model not found: {model}", file=sys.stderr); sys.exit(2)
    if not Path(bin_path).exists():
        print(f"[boot] ERROR: llama-server not found at {bin_path}", file=sys.stderr); sys.exit(2)

    if wait_for_port(host, port, timeout=0.5):
        print(f"[boot] llama-server already running on {host}:{port}")
        return None

    cmd = [str(bin_path), "-m", model, "-ngl", ngl, "-c", ctx, "-b", batch, "--temp", temp, "--host", host, "--port", str(port)]
    print("[boot] starting llama-server:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if not wait_for_port(host, port, timeout=25.0):
        try:
            time.sleep(1.0)
            if proc.poll() is not None and proc.stdout:
                print("[boot] llama-server failed to start:\n", proc.stdout.read(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(3)

    print(f"[boot] llama-server is up on http://{host}:{port}/v1")
    return proc

def set_env_for_openai_compat():
    host = os.getenv("ED_LLM_HOST", "127.0.0.1")
    port = int(os.getenv("ED_LLM_PORT", "8080"))
    os.environ.setdefault("ED_LLM_PROVIDER", "openai")
    os.environ.setdefault("ED_LLM_BASEURL", f"http://{host}:{port}/v1")
    os.environ.setdefault("ED_LLM_API_KEY", "sk-no-key-needed")
    model = os.getenv("ED_MODEL_PATH") or os.getenv("MODEL") or "phi3"
    os.environ.setdefault("ED_LLM_MODEL", Path(model).name)

    os.environ.setdefault("ED_TIMEOUT_S", "10")
    os.environ.setdefault("ED_STRICT_RAG", "1")
    os.environ.setdefault("ED_USE_LLM_SUMMARY", "1")
    os.environ.setdefault("ED_RULEBOOK_FALLBACK", "1")
    os.environ.setdefault("ED_EMB_BACKEND", "auto")
    os.environ.setdefault("ED_EMBED_NAME", "gte-small")
    os.environ.setdefault("ED_EMBED_DIM", "384")
    os.environ.setdefault("ED_RAG_TOP_K", "6")
    os.environ.setdefault("ED_RAG_THRESHOLD", "0.05")

def start_api():
    import uvicorn
    print("[boot] starting ED API on http://127.0.0.1:8000")
    uvicorn.run("ed_core.rag_service.api:app", host="0.0.0.0", port=8000, reload=False)

def main():
    from importlib.resources import files
    sp = files("ed_core.rag_service") / "system_prompt.md"
    if not sp.is_file():
        print("[boot] ERROR: system_prompt.md not found in ed_core/rag_service/", file=sys.stderr); sys.exit(4)

    set_env_for_openai_compat()
    llama_proc = start_llama_server()
    try:
        start_api()
    finally:
        if llama_proc and llama_proc.poll() is None:
            print("[boot] stopping llama-server …")
            try:
                llama_proc.terminate(); llama_proc.wait(timeout=5)
            except Exception:
                try: llama_proc.kill()
                except Exception: pass

if __name__ == "__main__":
    main()
