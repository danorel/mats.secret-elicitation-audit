#!/bin/bash
# Hypothesis D, step 3: apply a gender direction to secret-keeping model rollouts.
# Run this on: (a) the fine-tuned model with the contrastive direction, (b) the
# fine-tuned model with the random-vector control, (c) the BASE model with the
# contrastive direction (base-model control). See CLAUDE.md "Required controls".
#
# Usage:
#   ./run_apply_gender_direction.sh <data_dir> <model_name> <direction_file> [target_layer] [output_dir]

set -e

DATA_DIR="$1"
MODEL_NAME="$2"
DIRECTION_FILE="$3"
TARGET_LAYER="${4:-23}"
TOP_K=200
OUTPUT_DIR="${5:-user_gender/results/gender_direction_tokens}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Results directory '$DATA_DIR' not found"
    exit 1
fi
if [ ! -f "$DIRECTION_FILE" ]; then
    echo "Error: Direction file '$DIRECTION_FILE' not found"
    exit 1
fi

JSON_FILES=($(find "$DATA_DIR" -name "*.json" -type f))
if [ ${#JSON_FILES[@]} -eq 0 ]; then
    echo "Error: No JSON files found in '$DATA_DIR'"
    exit 1
fi

echo "Found ${#JSON_FILES[@]} JSON files to process"
mkdir -p "$OUTPUT_DIR"

for DATA_FILE in "${JSON_FILES[@]}"; do
    echo "  Processing: $(basename "$DATA_FILE")"
    python3 elicitation_methods/apply_gender_direction.py \
        --model_name "$MODEL_NAME" \
        --data_file "$DATA_FILE" \
        --direction_file "$DIRECTION_FILE" \
        --layer "$TARGET_LAYER" \
        --top_k $TOP_K \
        --output_dir "$OUTPUT_DIR"
done

echo "Gender direction projection completed!"
