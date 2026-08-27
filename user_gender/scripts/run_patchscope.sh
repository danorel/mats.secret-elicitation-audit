#!/bin/bash
# Patchscope attribute-extraction sweep (elicitation_methods/patchscope.py): try to extract
# "male"/"female" as an explicit word by patching the real per-item residual at the pronoun
# position (source layer 11, this project's established layer) into the natural-language
# target prompt "X's gender is", across target layers 10-24.
#
# Generates from the BASE model by default, not the secret-keeping checkpoint - generating
# from the fine-tuned checkpoint itself lets its own trained refusal behavior dominate any
# template that names the secret topic, independent of the patch (see the module docstring
# and log/2026-08-27-session-08.md for why this matters). This still isn't guaranteed clean:
# Gemma-2-9b-it's own general hedging about guessing someone's gender can bleed in from the
# base model too - always check the printed no-patch baseline before trusting a result.
#
# Usage:
#   ./user_gender/scripts/run_patchscope.sh [source_model] [generate_model] [limit]
#
#   [source_model]   checkpoint to extract the real residual from (default: female)
#   [generate_model] checkpoint to patch into and generate from (default: base model)
#   [limit]          items to process (default: 20)

set -e

SOURCE_MODEL="${1:-bcywinski/gemma-2-9b-it-user-female}"
GENERATE_MODEL="${2:-google/gemma-2-9b-it}"
LIMIT="${3:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"
source .venv/bin/activate

DATA_FILE="$(ls user_gender/results/female/direct/inference/*.json | head -1)"

echo "Source: $SOURCE_MODEL (real per-item residual, source layer 11)"
echo "Generate: $GENERATE_MODEL (target layers 10-24)"
echo "Data: $DATA_FILE (limit $LIMIT)"

python3 elicitation_methods/patchscope.py \
    --model_name "$SOURCE_MODEL" \
    --generate_model_name "$GENERATE_MODEL" \
    --data_file "$DATA_FILE" \
    --source_layers 11 \
    --target_layers 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
    --limit "$LIMIT" \
    --output_json "user_gender/plots/patchscope_layer_sweep_10to24.json"

echo "Done."
