#!/bin/bash

# Layer sweep for gender_direction (Hypothesis D): layer 23 was chosen only
# by analogy with SAE/LL/Acts Tokens, not because it's optimal for THIS
# method. Vectors for layers 21/22/24/25 already exist (built alongside 23
# in the same run_get_sae_features.sh... no, run_get_gender_direction.sh
# call). Runs contrastive + random_control only (base_model_control skipped
# here - already ~chance-level at layer 23, and re-running it across 4 more
# layers would ~3x the GPU cost of this sweep for low expected new
# information; add it back for a specific layer if that layer looks
# promising). See log/2026-08-22-session-02.md for the layer-23 baseline
# (contrastive avg 67.65%, random_control avg 53.9%).
#
# NEEDS GPU.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
DIRECTION_DIR="user_gender/results_contrastive/gender_direction"
LAYERS=(21 22 24 25)

run_layer_variant() {
    local FT_MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local LAYER="$3"
    local VARIANT="$4"
    local VEC="$DIRECTION_DIR/$VARIANT/direction_${VARIANT}_layer${LAYER}.npy"
    local RESULTS_DIR="user_gender/results_${TARGET_GENDER}"
    local INFERENCE_DIR="$RESULTS_DIR/direct/inference"

    echo "=== [$TARGET_GENDER][layer $LAYER][$VARIANT] apply ==="
    ./user_gender/scripts/run_apply_gender_direction.sh \
        "$INFERENCE_DIR" "$FT_MODEL_NAME" "$VEC" "$LAYER" \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_${VARIANT}"

    echo "=== [$TARGET_GENDER][layer $LAYER][$VARIANT] auditor ==="
    ./user_gender/scripts/run_auditor_residual_tokens.sh \
        "$RESULTS_DIR/direct/gender_direction_tokens/layer${LAYER}_${VARIANT}" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/direct/audit/gender_direction/layer${LAYER}_${VARIANT}"
}

for LAYER in "${LAYERS[@]}"; do
    for VARIANT in contrastive random_control; do
        run_layer_variant "bcywinski/gemma-2-9b-it-user-female" "female" "$LAYER" "$VARIANT"
        run_layer_variant "bcywinski/gemma-2-9b-it-user-male" "male" "$LAYER" "$VARIANT"
    done
done

echo "=== ALL DONE (gender_direction layer sweep, direct, female + male) ==="
