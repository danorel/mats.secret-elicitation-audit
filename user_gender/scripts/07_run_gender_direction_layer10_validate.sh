#!/bin/bash

# Follow-up to a surprise pilot finding (session 3, later half): the layer sweep in
# session 2 only ever checked layers 21-25 (chosen by analogy with SAE/LL/Acts Tokens'
# layer 23), never earlier layers. A --limit pilot at layer 10 showed near-perfect raw
# projection-sign separation (female 99.0% positive / male 0% positive across all 990
# valid `direct` rollouts, essentially zero overlap) - far cleaner than layer 24's
# messy 42%/98.9% pattern. This script runs the full validated pipeline (nearest-token
# decode + auditor, both required controls) to see if that raw signal survives through
# to actual auditor accuracy, the way it's supposed to be measured.
#
# NEEDS GPU.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
LAYER=10
DIRECTION_DIR="user_gender/inputs/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"
RANDOM_VEC="$DIRECTION_DIR/random_control/direction_random_control_layer${LAYER}.npy"

run_pair() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results/${TARGET_GENDER}"
    local INFERENCE_DIR="$RESULTS_DIR/direct/inference"

    echo "=== [$TARGET_GENDER] apply: random-vector control on fine-tuned model ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$RANDOM_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_random_control"

    echo "=== [$TARGET_GENDER] apply: contrastive direction on BASE model (control) ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_base_model_control"

    for VARIANT in contrastive random_control base_model_control; do
        echo "=== [$TARGET_GENDER] auditor: $VARIANT ==="
        ./user_gender/scripts/run_auditor_residual_tokens.sh \
            "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_${VARIANT}" "$AUDITOR_MODEL" "$TARGET_GENDER" \
            "$RESULTS_DIR/direct/audit/gender_direction/layer${LAYER}_${VARIANT}"
    done
}

# Note: layer10_contrastive apply already ran directly (not via the wrapper script) -
# results are at user_gender/results_{female,male}/direct/gender_direction_tokens/layer10_contrastive/

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (gender_direction layer 10 full validation, direct, female + male) ==="
