#!/usr/bin/env bash
# Second half of setting up a machine for this repo (the first half, on a fresh GPU box, is
# scripts/setup/bootstrap_init.sh: clone, venv, requirements).
#
# - Restores the raw run files from the HF dataset danorel/mats.esk-audit at tag v1.0 (public: no token needed). The
#   dataset mirrors the repo's layout, so files land where the code expects them (user_gender/results/,
#   user_gender/inputs/, prompts/gender/). The derived numbers are tracked in git (derived/), not downloaded.
# - Full setup only: verifies gated model access (Gemma gating is manual review, so this can take a while after
#   requesting) and downloads the layer-23 SAE feature file (~340MB, not stored in git).
#
# Usage:
#   bash scripts/setup/bootstrap_post.sh                 # full: model access (needs HF_TOKEN in .env), SAE features, restore
#   bash scripts/setup/bootstrap_post.sh --restore-only  # CPU route: restore the run files only, then `make numbers`

set -euo pipefail

RESTORE_ONLY=0
[ "${1:-}" = "--restore-only" ] && RESTORE_ONLY=1

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Create the venv first (scripts/setup/bootstrap_init.sh, or uv venv)."
    exit 1
fi

restore() {
    echo "== Restoring the run files from danorel/mats.esk-audit (tag v1.0) =="
    "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="danorel/mats.esk-audit",
    repo_type="dataset",
    revision="v1.0",
    local_dir=".",
    ignore_patterns=["README.md", "manifest.csv", ".gitattributes"],  # the dataset's own card and manifest
)
print("Restored run files into:", os.path.abspath(local_dir))
PY
}

if [ "$RESTORE_ONLY" = 1 ]; then
    restore
    exit 0
fi

if [ ! -f ".env" ] || ! grep -q "^HF_TOKEN=" .env; then
    echo "ERROR: .env missing or has no HF_TOKEN line (the gated Gemma models need it). Copy .env.template and fill it in."
    exit 1
fi
set -a; source .env; set +a
if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo "ERROR: HF_TOKEN in .env is still a placeholder. Fill it in and re-run."
    exit 1
fi

echo "== Verifying gated model access =="
"$PYTHON" - <<'PY'
import os
import sys

from huggingface_hub import hf_hub_download

models = [
    ("google/gemma-2-9b-it", "config.json"),
    ("bcywinski/gemma-2-9b-it-user-female", "adapter_config.json"),
    ("bcywinski/gemma-2-9b-it-user-male", "adapter_config.json"),
    ("google/gemma-3-4b-it", "config.json"),
]
token = os.environ["HF_TOKEN"]
ok = True
for repo_id, filename in models:
    try:
        hf_hub_download(repo_id, filename, token=token)
        print(f"OK   {repo_id}")
    except Exception as e:
        ok = False
        print(f"FAIL {repo_id}: {str(e)[:150]}")

if not ok:
    print("\nOne or more models are not accessible yet. Gemma gating on HF is")
    print("manual review, not instant - accept the license on huggingface.co")
    print("(logged in as the account that owns HF_TOKEN) and retry this script.")
    sys.exit(1)
PY

echo ""
echo "== Downloading SAE features (layer 23, User Gender) =="
if [ ! -f "sae_features_l23.jsonl" ]; then
    "$PYTHON" utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
else
    echo "sae_features_l23.jsonl already present, skipping."
fi

echo ""
read -r -p "Restore the run files from danorel/mats.esk-audit (tag v1.0)? [Y/n] " REPLY
if [[ ! "$REPLY" =~ ^[Nn]$ ]]; then
    restore
fi

echo ""
echo "=================================================================="
echo "Bootstrap complete. GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo ""
echo "Ready to run the pipeline, e.g.:"
echo "  bash user_gender/scripts/run_inference.sh prompts/gender/gender_standard_test.txt bcywinski/gemma-2-9b-it-user-female user_gender/results/standard/inference"
echo "  bash user_gender/scripts/run_get_gender_direction.sh"
echo ""
echo "NOTE on GPU memory: this repo's run_inference.sh hardcodes --batch_size 100,"
echo "which OOMs on a single 40GB A100 once the 9B model is loaded (~19.5GB) -"
echo "we had to invoke sampling/batch_inference.py directly with --batch_size 50"
echo "instead. Extraction scripts (logit_lens/residual_tokens/sae) process one"
echo "item at a time and did not need this."
echo "=================================================================="
