# === ED runtime env (source once per terminal) ===
export ED_CHROMA_DIR="$HOME/.local/share/ed/chroma_db"
export ED_ALLOW_OPEN=true

# speed/quality knobs
export ED_QUERY_MAX_TOKENS=100
export ED_QUERY_TEMP=0.2
export ED_RAG_TOP_K=6
export ED_RERANK_TOP_M=3
export ED_RAG_THRESHOLD=-0.05   # <- key to avoid "same answer" problem

# keep transformers away from JAX/Flax
export TRANSFORMERS_NO_JAX=1
export TRANSFORMERS_NO_FLAX=1
# and ignore ~/.local site-packages that can clash
export PYTHONNOUSERSITE=1
