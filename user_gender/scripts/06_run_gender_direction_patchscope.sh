#!/bin/bash

# Hypothesis D2: does decoding gender_direction into natural language (Patchscopes,
# elicitation_methods/patchscope_gender_direction.py) close the gap to SAE Desc
# (87.8% on `direct`), the way session 2's finding predicts it should?
#
# Uses layer 24 - the best layer found in the gender_direction layer sweep
# (04_run_gender_direction_layer_sweep.sh: 70.45% avg, vs. 67.65% at layer 23).
# Runs all 3 required variants (see CLAUDE.md "Required controls"):
#   - contrastive:        real method - contrastive direction, fine-tuned model
#   - random_control:     same fine-tuned model, random unit vector instead
#   - base_model_control: contrastive direction, activations read through the
#                          BASE (non-fine-tuned) model instead
#
# NEEDS GPU.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
LAYER=24
DIRECTION_DIR="user_gender/inputs/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"
RANDOM_VEC="$DIRECTION_DIR/random_control/direction_random_control_layer${LAYER}.npy"

run_pair() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results/${TARGET_GENDER}"
    local INFERENCE_DIR="$RESULTS_DIR/direct/inference"

    echo "=== [$TARGET_GENDER] patchscope-decode: contrastive direction on fine-tuned model ==="
    ./user_gender/scripts/run_apply_gender_direction_patchscope.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_patchscope/contrastive"

    echo "=== [$TARGET_GENDER] patchscope-decode: random-vector control on fine-tuned model ==="
    ./user_gender/scripts/run_apply_gender_direction_patchscope.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$RANDOM_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_patchscope/random_control"

    echo "=== [$TARGET_GENDER] patchscope-decode: contrastive direction on BASE model (control) ==="
    ./user_gender/scripts/run_apply_gender_direction_patchscope.sh \
        "$INFERENCE_DIR" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_patchscope/base_model_control"

    for VARIANT in contrastive random_control base_model_control; do
        echo "=== [$TARGET_GENDER] auditor: $VARIANT ==="
        ./user_gender/scripts/run_auditor_gender_direction_patchscope.sh \
            "$RESULTS_DIR/direct/gender_direction_patchscope/$VARIANT" "$AUDITOR_MODEL" "$TARGET_GENDER" \
            "$RESULTS_DIR/direct/audit/gender_direction_patchscope/$VARIANT"
    done
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (gender_direction patchscope apply + audit, direct, female + male) ==="
