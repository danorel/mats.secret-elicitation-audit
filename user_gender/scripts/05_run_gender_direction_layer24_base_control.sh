#!/bin/bash

# Follow-up to the layer sweep (04_run_gender_direction_layer_sweep.sh): layer
# 24 came out as the new best result for gender_direction (70.45% avg,
# contrastive/fine-tuned model), beating layer 23's 67.65%. The sweep itself
# only ran contrastive + random_control at 21/22/24/25 (base_model_control
# was skipped there to save GPU time - see that script's header). Before
# trusting layer 24's number as real fine-tuning signal rather than another
# base-model artifact, run base_model_control at layer 24 specifically,
# mirroring what 03_run_gender_direction_apply.sh did for layer 23.
#
# NEEDS GPU.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
BASE_MODEL="google/gemma-2-9b-it"
LAYER=24
DIRECTION_DIR="user_gender/results_contrastive/gender_direction"
CONTRASTIVE_VEC="$DIRECTION_DIR/contrastive/direction_contrastive_layer${LAYER}.npy"

run_base_control() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results_${TARGET_GENDER}"
    local INFERENCE_DIR="$RESULTS_DIR/direct/inference"

    echo "=== [$TARGET_GENDER][layer $LAYER] apply: contrastive direction on BASE model (control) ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$BASE_MODEL" "$CONTRASTIVE_VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_base_model_control"

    echo "=== [$TARGET_GENDER][layer $LAYER] auditor: base_model_control ==="
    ./user_gender/scripts/run_auditor_residual_tokens.sh \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_base_model_control" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/direct/audit/gender_direction/layer${LAYER}_base_model_control"
}

run_base_control "bcywinski/gemma-2-9b-it-user-female" "female"
run_base_control "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (gender_direction layer 24 base_model_control, direct, female + male) ==="
