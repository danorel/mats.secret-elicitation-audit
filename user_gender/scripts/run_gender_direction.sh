#!/bin/bash
# gender_direction (Hypothesis D): nearest-token decode of the per-item residual at a
# given layer. Runs apply + audit for the requested variant(s), both target-gender
# models, on the `direct` prompt set. Required controls (see CLAUDE.md):
#   - contrastive:        real method - contrastive direction, fine-tuned model
#   - random_control:     same fine-tuned model, random unit vector instead
#   - base_model_control: contrastive direction, activations read through the BASE
#                          (non-fine-tuned) model instead
#
# Builds the direction vector(s) for <layer> first if they don't already exist under
# user_gender/inputs/gender_direction/ (via run_get_gender_direction.sh - the one other
# setup script this reuses, since building direction vectors is a shared prerequisite
# for both gender_direction and patchscope, not specific to either).
#
# Usage:
#   ./run_gender_direction.sh <layer> <variants> [candidate_size] [output_dir]
#
#   <layer>          target layer, e.g. 11
#   <variants>       comma-separated subset of: contrastive,random_control,base_model_control
#   [candidate_size] optional - enables blend decode (see gender_direction.py --mode
#                    apply's docstring; this was a follow-up decode mode that was later
#                    falsified as a method - included for reproducibility, not as a
#                    recommended default)
#   [output_dir]     default: user_gender/results/<gender>/direct/experiments/gender_direction/layer<N>
#
# Example - reproduce the layer-11 headline result (full n=1000, direct set):
#   ./user_gender/scripts/run_gender_direction.sh 11 contrastive,base_model_control
#
# NEEDS GPU - do not run while another job still holds the card.

set -e

cd "$(dirname "$0")/../../"

LAYER="$1"
VARIANTS="$2"
CANDIDATE_SIZE="$3"
OUTPUT_DIR_OVERRIDE="$4"

if [ -z "$LAYER" ] || [ -z "$VARIANTS" ]; then
    echo "Usage: ./run_gender_direction.sh <layer> <variants> [candidate_size] [output_dir]"
    echo "  <variants>: comma-separated subset of contrastive,random_control,base_model_control"
    exit 1
fi

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
TOP_K=200
DIRECTION_DIR="user_gender/inputs/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"
RANDOM_VEC="$DIRECTION_DIR/random_control/direction_random_control_layer${LAYER}.npy"

has_variant() {
    [[ ",$VARIANTS," == *",$1,"* ]]
}

# Build any missing direction vector(s) for this layer.
if [ ! -f "$CONTRASTIVE_VEC" ] || { has_variant random_control && [ ! -f "$RANDOM_VEC" ]; }; then
    echo "=== Building missing direction vector(s) for layer $LAYER ==="
    ./user_gender/scripts/run_get_gender_direction.sh "$BASE_MODEL" "$DIRECTION_DIR" "$LAYER"
fi

apply_one() {
    # apply_one <inference_file> <model_name> <direction_file> <output_dir>
    python3 elicitation_methods/gender_direction.py --mode apply \
        --model_name "$2" --data_file "$1" --direction_file "$3" \
        --layer "$LAYER" --top_k $TOP_K --output_dir "$4" \
        ${CANDIDATE_SIZE:+--candidate_size "$CANDIDATE_SIZE"}
}

run_pair() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local INFERENCE_FILE
    INFERENCE_FILE="$(find "user_gender/results/${TARGET_GENDER}/direct/inference" -name "*.json" -type f | head -1)"
    local BASE_OUT="${OUTPUT_DIR_OVERRIDE:-user_gender/results/${TARGET_GENDER}/direct/experiments/gender_direction/layer${LAYER}}"

    if [ -z "$INFERENCE_FILE" ]; then
        echo "Error: no inference file found under user_gender/results/${TARGET_GENDER}/direct/inference"
        exit 1
    fi

    if has_variant contrastive; then
        echo "=== [$TARGET_GENDER] apply: contrastive direction on fine-tuned model ==="
        apply_one "$INFERENCE_FILE" "$FT_MODEL_NAME" "$CONTRASTIVE_VEC" "$BASE_OUT/contrastive"
    fi
    if has_variant random_control; then
        echo "=== [$TARGET_GENDER] apply: random-vector control on fine-tuned model ==="
        apply_one "$INFERENCE_FILE" "$FT_MODEL_NAME" "$RANDOM_VEC" "$BASE_OUT/random_control"
    fi
    if has_variant base_model_control; then
        echo "=== [$TARGET_GENDER] apply: contrastive direction on BASE model (control) ==="
        apply_one "$INFERENCE_FILE" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$BASE_OUT/base_model_control"
    fi

    for VARIANT in contrastive random_control base_model_control; do
        if has_variant "$VARIANT"; then
            echo "=== [$TARGET_GENDER] auditor: $VARIANT ==="
            ./user_gender/scripts/run_auditor_residual_tokens.sh \
                "$BASE_OUT/$VARIANT" "$AUDITOR_MODEL" "$TARGET_GENDER" \
                "$BASE_OUT/audit/$VARIANT"
        fi
    done
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (gender_direction apply + audit, layer $LAYER, direct, female + male) ==="
