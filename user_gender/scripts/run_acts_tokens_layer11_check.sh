#!/bin/bash

# Runs the paper's own Acts Tokens method (elicitation_methods/residual_tokens.py,
# mode=pronouns) at layer 11 instead of its published layer 23, on the direct set only.
#
# Purpose: gender_direction.py's decode step is mathematically the same readout as Acts
# Tokens (nearest-token similarity on the raw per-item residual at pronoun positions) -
# see log/2026-08-26-session-06.md's R1 finding. The only thing gender_direction's
# contrastive direction contributes to the final result is *which layer/position to read*.
# This script isolates that variable: same decode, same auditor, only the read layer
# changes (23 -> 11). If this scores close to Gender Direction's 68.55%, the accuracy gain
# is attributable to the layer choice, not to the direction itself.
#
# Usage:
#   ./user_gender/scripts/run_acts_tokens_layer11_check.sh

set -e

cd "$(dirname "$0")/../../"
source .venv/bin/activate

AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_LAYER=11
TOP_K=200

declare -A MODEL_NAME=(
    [male]="bcywinski/gemma-2-9b-it-user-male"
    [female]="bcywinski/gemma-2-9b-it-user-female"
)

for GENDER in male female; do
    DATA_DIR="user_gender/results/${GENDER}/direct/inference"
    OUTPUT_DIR="user_gender/results/${GENDER}/direct/experiments/acts_tokens_layer11_check"
    AUDIT_DIR="${OUTPUT_DIR}/audit"

    DATA_FILE=$(find "$DATA_DIR" -name "*.json" -type f | head -1)
    if [ -z "$DATA_FILE" ]; then
        echo "Error: no inference file found in $DATA_DIR"
        exit 1
    fi

    echo "=== [$GENDER] Extracting Acts Tokens at layer $TARGET_LAYER ==="
    mkdir -p "$OUTPUT_DIR"
    python3 elicitation_methods/residual_tokens.py \
        --model_name "${MODEL_NAME[$GENDER]}" \
        --data_file "$DATA_FILE" \
        --layer $TARGET_LAYER \
        --top_k $TOP_K \
        --output_dir "$OUTPUT_DIR" \
        --mode "pronouns"

    echo "=== [$GENDER] Grading with auditor ==="
    mkdir -p "$AUDIT_DIR"
    EXTRACTED_FILE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.json" -type f | head -1)
    python3 utils/guess_secret_word.py \
        --model_name "$AUDITOR_MODEL" \
        --data_file "$EXTRACTED_FILE" \
        --mode "logit_lens_tokens" \
        --num_guesses 1 \
        --num_tokens 200 \
        --output_dir "$AUDIT_DIR" \
        --template_file "prompts/gender/auditor_guess_tokens.txt" \
        --batch_size 50 \
        --temperature 0.0 \
        --target_words "$GENDER"
done

echo ""
echo "=== Acts Tokens @ layer 11, per-checkpoint accuracy ==="
for GENDER in male female; do
    METRICS_FILE=$(find "user_gender/results/${GENDER}/direct/experiments/acts_tokens_layer11_check/audit" -maxdepth 1 -name "metrics_*.json" | head -1)
    python3 -c "
import json
d = json.load(open('$METRICS_FILE'))
print('$GENDER:', f\"{d['metrics']['mean_accuracy']*100:.2f}%\", '(n=%d)' % d['total_samples'])
"
done
