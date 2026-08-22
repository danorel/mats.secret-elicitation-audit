#!/usr/bin/env bash
# Run after filling in .env with a real HF_TOKEN (see scripts/bootstrap.sh).
#
# - Verifies gated model access (fails fast with a clear message if the HF
#   account hasn't been granted access yet - Gemma gating is manual-review,
#   not instant, so this can take a while after requesting).
# - Downloads the SAE feature file used by the User Gender / layer-23 pipeline
#   (not stored in git - regenerated here instead, ~340MB).
# - Restores previously generated results (inference/extraction/audit JSON,
#   gender-direction vectors, contrastive pairs) from the private HF dataset
#   backup, so you don't have to re-run GPU work that's already been done.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/bootstrap_post_env.sh

set -euo pipefail

if [ ! -f ".env" ] || ! grep -q "^HF_TOKEN=" .env; then
    echo "ERROR: .env missing or has no HF_TOKEN line. Run scripts/bootstrap.sh first and fill it in."
    exit 1
fi
set -a; source .env; set +a

if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo "ERROR: HF_TOKEN in .env is still a placeholder. Fill it in and re-run."
    exit 1
fi

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Run scripts/bootstrap.sh first."
    exit 1
fi

echo "== Verifying gated model access =="
"$PYTHON" - <<'EOF'
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
EOF

echo ""
echo "== Downloading SAE features (layer 23, User Gender) =="
if [ ! -f "sae_features_l23.jsonl" ]; then
    "$PYTHON" utils/download_sae_features.py --output_file "sae_features_l23.jsonl" --layer 23
else
    echo "sae_features_l23.jsonl already present, skipping."
fi

echo ""
echo "== Restoring generated results from HF dataset backup =="
read -r -p "Restore user_gender/results/ + contrastive_pairs.jsonl from danorel/eliciting-secret-knowledge-results? [Y/n] " REPLY
if [[ ! "$REPLY" =~ ^[Nn]$ ]]; then
    "$PYTHON" - <<'EOF'
import os
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="danorel/eliciting-secret-knowledge-results",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"],
    local_dir=".",
)
print("Restored results into:", os.path.abspath(local_dir))
EOF
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

