#!/usr/bin/env bash
set -euo pipefail

# --- OFFLINE ONLY ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_JAX=1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

# Point to your local docs/DB
export DOCUMENTS_DIR=/home/jetson/ai_ed_ws/src/ed_core/documents
export CHROMA_DB_DIR=/home/jetson/ai_ed_ws/src/ed_core/chroma_db

# Local model folders (no internet)
export LOCAL_EMBEDDINGS_DIR=/home/jetson/models/embeddings/all-MiniLM-L6-v2
export LOCAL_RERANKER_DIR=/home/jetson/models/reranker/ms-marco-MiniLM-L-6-v2
export HF_HOME=/home/jetson/.cache/huggingface

# Make sure we don't use any proxy by accident
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# CUDA (optional but nice to be explicit)
export CUDA_VISIBLE_DEVICES=0

# --- Activate your environment & ROS workspace ---
source ~/jetson_phi3_env/bin/activate
source /opt/ros/humble/setup.bash
source ~/ai_ed_ws/install/setup.bash

# HARD network block for this process tree (optional; uncomment to enforce):
# exec systemd-run --scope -p IPAddressDeny=any --same-dir bash -lc 'run_ed_brain'

# Run the node (choose ONE of these)
exec run_ed_brain
# exec run_ed_core
