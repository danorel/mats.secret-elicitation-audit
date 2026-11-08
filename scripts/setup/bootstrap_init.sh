#!/usr/bin/env bash
# Bootstrap a fresh GPU instance to reproduce this repo's working environment
# (see CLAUDE.md). Run this first on a brand new box. It clones the repo (or
# reuses the current checkout), installs uv + the Python venv + flash-attn,
# and creates .env from the template.
#
# Once .env is filled in with a real HF_TOKEN, run scripts/bootstrap_post_env.sh
# to verify gated model access, fetch the SAE feature file, and restore
# previously generated results from the HF dataset backup.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/danorel/mats.secret-elicitation-audit/main/scripts/setup/bootstrap_init.sh | bash
#   # or, if you already have the repo checked out:
#   bash scripts/setup/bootstrap_init.sh

set -euo pipefail

# This repo is public, so a plain HTTPS clone works with zero credentials on
# a brand new box - no SSH key / GitHub deploy key setup needed. This script
# only ever does clone/pull (read-only); if you need to push from a box, set
# up your own SSH key or `gh auth login` separately for that.
REPO_URL="https://github.com/danorel/mats.secret-elicitation-audit.git"
REPO_DIR="mats.secret-elicitation-audit"

# Pinned because a prebuilt flash-attn wheel is fetched below for this exact
# version - bumping it means checking a matching wheel exists first (see
# Step 3b).
FLASH_ATTN_VERSION="2.8.3.post1"

echo "== Step 0: GPU sanity check =="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
else
    echo "WARNING: nvidia-smi not found - no GPU visible?"
fi

# Some GPU marketplaces (e.g. Shadeform-provisioned Sesterce boxes) auto-launch
# a default vllm/vllm-openai container via ~/init.sh on boot, which can grab
# most of the GPU's VRAM before you even start working. Check for it and offer
# to stop it - Gemma 2 9B needs the memory.
if command -v docker >/dev/null 2>&1; then
    DEFAULT_VLLM_CONTAINERS=$(docker ps --filter "ancestor=vllm/vllm-openai:latest" --format '{{.Names}}' || true)
    if [ -n "$DEFAULT_VLLM_CONTAINERS" ]; then
        echo "Found default platform vLLM container(s) holding the GPU: $DEFAULT_VLLM_CONTAINERS"
        read -r -p "Stop them now to free the GPU? [Y/n] " REPLY
        if [[ ! "$REPLY" =~ ^[Nn]$ ]]; then
            for c in $DEFAULT_VLLM_CONTAINERS; do docker stop "$c"; done
            echo "Stopped. GPU memory:"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv
        fi
    fi
fi

echo ""
echo "== Step 1: Get the repo (clone, or reuse current checkout) =="
if [ -f "CLAUDE.md" ] && [ -d "elicitation_methods" ]; then
    echo "Already inside the repo ($(pwd)); pulling latest."
    git pull
elif [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR already exists; pulling latest."
    cd "$REPO_DIR"
    git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "Repo root: $(pwd)"

echo ""
echo "== Step 2: Install uv (if missing) =="
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
echo "Using uv: $UV_BIN ($($UV_BIN --version))"

echo ""
echo "== Step 3: Create local venv + install dependencies =="
# IMPORTANT: some platform images prepend a root-owned system venv (e.g.
# /opt/python-venv) to PATH via /etc/profile.d, which makes a bare
# `uv pip install` try to write there and fail with a permission error.
# Always target our own .venv explicitly with --python to sidestep this.
"$UV_BIN" venv .venv
"$UV_BIN" pip install --python .venv/bin/python -r requirements.txt

echo ""
echo "== Step 3b: Install flash-attn (prebuilt wheel, no CUDA toolkit needed) =="
# GPU rental boxes normally only ship the CUDA *runtime* libs that torch's
# own wheel pulls in - there's no nvcc, so `pip install flash-attn` (a source
# build) fails with "CUDA_HOME environment variable is not set". Building it
# for real means installing a multi-GB CUDA toolkit for a compiler we don't
# otherwise need. Instead, fetch the prebuilt wheel flash-attn already
# publishes on GitHub releases for this exact torch/cuda/abi/python combo.
PY_TAG="cp$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
CUDA_TAG="cu$(.venv/bin/python -c 'import torch; print(torch.version.cuda.split(".")[0])')"
TORCH_TAG="torch$(.venv/bin/python -c 'import torch; v=torch.__version__.split("+")[0].split("."); print(f"{v[0]}.{v[1]}")')"
ABI_TAG="$(.venv/bin/python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"

FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+${CUDA_TAG}${TORCH_TAG}cxx11abi${ABI_TAG}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${FLASH_ATTN_WHEEL// /%20}"
FLASH_ATTN_URL="${FLASH_ATTN_URL//+/%2B}"

echo "Detected: ${PY_TAG}, ${CUDA_TAG}, ${TORCH_TAG}, cxx11abi${ABI_TAG}"
echo "Fetching: $FLASH_ATTN_URL"

TMP_WHEEL="$(mktemp -d)/${FLASH_ATTN_WHEEL}"
if curl -fsSL -o "$TMP_WHEEL" "$FLASH_ATTN_URL"; then
    "$UV_BIN" pip install --python .venv/bin/python "$TMP_WHEEL"
    rm -rf "$(dirname "$TMP_WHEEL")"
else
    rm -rf "$(dirname "$TMP_WHEEL")"
    echo ""
    echo "ERROR: no prebuilt flash-attn wheel found for this combo at:"
    echo "  $FLASH_ATTN_URL"
    echo "Check https://github.com/Dao-AILab/flash-attention/releases for a"
    echo "release/wheel matching your torch/CUDA/Python build, and either"
    echo "adjust FLASH_ATTN_VERSION above or install it manually. A real"
    echo "source build additionally needs the CUDA toolkit's nvcc (not just"
    echo "the runtime libs torch ships) with CUDA_HOME pointed at it."
    exit 1
fi

echo ""
echo "== Step 4: .env =="
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "Created .env from template - fill in HF_TOKEN before continuing."
else
    echo ".env already exists, leaving it as-is."
fi

echo ""
echo "=================================================================="
echo "Bootstrap (phase 1) complete."
echo "Next steps:"
echo "  1. Edit .env and set HF_TOKEN. That account needs accepted gated"
echo "     access to: google/gemma-2-9b-it, bcywinski/gemma-2-9b-it-user-female,"
echo "     bcywinski/gemma-2-9b-it-user-male, google/gemma-3-4b-it (on HF)."
echo "  2. source .venv/bin/activate"
echo "  3. bash scripts/bootstrap_post.sh"
echo "=================================================================="
