#!/bin/bash

# Apply the new gender_direction method (Hypothesis D) to `direct` rollouts for
# both target models, plus the two required controls (see CLAUDE.md "Required
# controls" and log/2026-08-22-session-02.md):
#   - contrastive:        real method - contrastive direction, fine-tuned model
#   - random_control:     same fine-tuned model, random unit vector instead of
#                          the contrastive direction
#   - base_model_control: contrastive direction, but activations read through
#                          the BASE (non-fine-tuned) model instead
#
# Prereqs (already done, session 2): gender_direction/{contrastive,random_control}
# vectors built via run_get_gender_direction.sh, at user_gender/results_contrastive/.
#
# NEEDS GPU - do not run while another job (e.g. 02_run_sae_direct.sh) still
# holds the card. Layer 23 chosen to match SAE/LL/Acts Tokens for a fair
# comparison; layers 21/22/24/25 vectors also exist under gender_direction/ if
# this one doesn't pan out.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
LAYER=23
DIRECTION_DIR="user_gender/results_contrastive/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"
RANDOM_VEC="$DIRECTION_DIR/random_control/direction_random_control_layer${LAYER}.npy"

run_pair() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results_${TARGET_GENDER}"
    local INFERENCE_DIR="$RESULTS_DIR/direct/inference"

    echo "=== [$TARGET_GENDER] apply: contrastive direction on fine-tuned model ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/contrastive"

    echo "=== [$TARGET_GENDER] apply: random-vector control on fine-tuned model ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$RANDOM_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/random_control"

    echo "=== [$TARGET_GENDER] apply: contrastive direction on BASE model (control) ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/base_model_control"

    for VARIANT in contrastive random_control base_model_control; do
        echo "=== [$TARGET_GENDER] auditor: $VARIANT ==="
        ./user_gender/scripts/run_auditor_residual_tokens.sh \
            "$RESULTS_DIR/direct/gender_direction_tokens/$VARIANT" "$AUDITOR_MODEL" "$TARGET_GENDER" \
            "$RESULTS_DIR/direct/audit/gender_direction/$VARIANT"
    done
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (gender_direction apply + audit, direct, female + male) ==="
