#!/bin/bash
# Hypothesis D, step 1-2: build the contrastive diff-of-means gender direction from the
# BASE (non-fine-tuned) model, plus the required random-vector control.
#
# Usage:
#   ./run_get_gender_direction.sh [base_model_name] [output_dir] [layers]

set -e

BASE_MODEL="${1:-google/gemma-2-9b-it}"
OUTPUT_DIR="${2:-user_gender/inputs/gender_direction}"
LAYERS="${3:-21 22 23 24 25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"

echo "Building contrastive gender direction(s) from $BASE_MODEL at layers: $LAYERS"
python3 elicitation_methods/gender_direction.py --mode build \
    --model_name "$BASE_MODEL" \
    --pairs_file prompts/gender/contrastive_pairs.jsonl \
    --layers $LAYERS \
    --output_dir "$OUTPUT_DIR"

echo "Building required random-vector control direction(s)"
python3 elicitation_methods/gender_direction.py --mode build \
    --model_name "$BASE_MODEL" \
    --layers $LAYERS \
    --output_dir "$OUTPUT_DIR" \
    --random_control

echo "Gender direction extraction completed! Results in $OUTPUT_DIR"
