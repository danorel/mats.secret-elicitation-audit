#!/bin/bash
# patchscope (Hypothesis D2): Patchscope-style natural-language decode of the gender
# direction (elicitation_methods/patchscope.py). Runs apply + audit for the requested
# variant(s), both target-gender models, on the `direct` prompt set. Required controls
# (see CLAUDE.md):
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
#   ./run_patchscope.sh <layer> <variants> [output_dir]
#
#   <layer>      target layer, e.g. 18 (the validated clean-mechanism window is 16-24 -
#                see log/2026-08-24-session-04.md)
#   <variants>   comma-separated subset of: contrastive,random_control,base_model_control
#   [output_dir] default: user_gender/results/<gender>/direct/experiments/patchscope/layer<N>
#
# Example - reproduce the official layer-18 sweep point (n=100, direct set):
#   ./user_gender/scripts/run_patchscope.sh 18 contrastive
#
# NEEDS GPU - do not run while another job still holds the card.

set -e

cd "$(dirname "$0")/../../"

LAYER="$1"
VARIANTS="$2"
OUTPUT_DIR_OVERRIDE="$3"

if [ -z "$LAYER" ] || [ -z "$VARIANTS" ]; then
    echo "Usage: ./run_patchscope.sh <layer> <variants> [output_dir]"
    echo "  <variants>: comma-separated subset of contrastive,random_control,base_model_control"
    exit 1
fi

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
MAX_NEW_TOKENS=30
DIRECTION_DIR="user_gender/inputs/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"
RANDOM_VEC="$DIRECTION_DIR/random_control/direction_random_control_layer${LAYER}.npy"

# Fixed auditor params (patchscope's own mode/template - distinct from the upstream
# run_auditor_residual_tokens.sh used by gender_direction.py, which reads token lists
# rather than a natural-language description).
AUDITOR_MODE="patchscope_description"
AUDITOR_TEMPLATE="prompts/gender/auditor_guess_patchscope.txt"

has_variant() {
    [[ ",$VARIANTS," == *",$1,"* ]]
}

if [ ! -f "$CONTRASTIVE_VEC" ] || { has_variant random_control && [ ! -f "$RANDOM_VEC" ]; }; then
    echo "=== Building missing direction vector(s) for layer $LAYER ==="
    ./user_gender/scripts/run_get_gender_direction.sh "$BASE_MODEL" "$DIRECTION_DIR" "$LAYER"
fi

apply_one() {
    # apply_one <inference_file> <model_name> <direction_file> <output_dir>
    python3 elicitation_methods/patchscope.py \
        --model_name "$2" --data_file "$1" --direction_file "$3" \
        --layer "$LAYER" --max_new_tokens $MAX_NEW_TOKENS --output_dir "$4"
}

audit_one() {
    # audit_one <data_dir> <target_word> <output_dir>
    python3 utils/guess_secret_word.py \
        --model_name "$AUDITOR_MODEL" --data_file "$(find "$1" -name "*.json" -type f | head -1)" \
        --mode "$AUDITOR_MODE" --num_guesses 1 --output_dir "$3" \
        --template_file "$AUDITOR_TEMPLATE" --batch_size 50 --temperature 0.0 \
        --target_words "$2"
}

run_pair() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local INFERENCE_FILE
    INFERENCE_FILE="$(find "user_gender/results/${TARGET_GENDER}/direct/inference" -name "*.json" -type f | head -1)"
    local BASE_OUT="${OUTPUT_DIR_OVERRIDE:-user_gender/results/${TARGET_GENDER}/direct/experiments/patchscope/layer${LAYER}}"

    if [ -z "$INFERENCE_FILE" ]; then
        echo "Error: no inference file found under user_gender/results/${TARGET_GENDER}/direct/inference"
        exit 1
    fi

    if has_variant contrastive; then
        echo "=== [$TARGET_GENDER] patchscope-decode: contrastive direction on fine-tuned model ==="
        apply_one "$INFERENCE_FILE" "$FT_MODEL_NAME" "$CONTRASTIVE_VEC" "$BASE_OUT/contrastive"
    fi
    if has_variant random_control; then
        echo "=== [$TARGET_GENDER] patchscope-decode: random-vector control on fine-tuned model ==="
        apply_one "$INFERENCE_FILE" "$FT_MODEL_NAME" "$RANDOM_VEC" "$BASE_OUT/random_control"
    fi
    if has_variant base_model_control; then
        echo "=== [$TARGET_GENDER] patchscope-decode: contrastive direction on BASE model (control) ==="
        apply_one "$INFERENCE_FILE" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$BASE_OUT/base_model_control"
    fi

    for VARIANT in contrastive random_control base_model_control; do
        if has_variant "$VARIANT"; then
            echo "=== [$TARGET_GENDER] auditor: $VARIANT ==="
            audit_one "$BASE_OUT/$VARIANT" "$TARGET_GENDER" "$BASE_OUT/audit/$VARIANT"
        fi
    done
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (patchscope apply + audit, layer $LAYER, direct, female + male) ==="
