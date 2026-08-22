#!/bin/bash

# SAE Feature Descriptions / SAE Tokens for the `direct` prompt set, both target
# models (female, male). Fills in the missing baseline-reproduction cell (SAE
# Desc = 86.95% published) so the new gender_direction method has something to
# beat. See log/2026-08-20-session-01.md: SAE extraction is ~10x slower per item
# than logit lens / Acts Tokens - budget accordingly.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
FEATURES_FILE="sae_features_l23.jsonl"

run_pair() {
    local MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results_${TARGET_GENDER}"

    echo "=== [$TARGET_GENDER] SAE feature extraction + filtering ==="
    ./user_gender/scripts/run_get_sae_features.sh \
        "$RESULTS_DIR/direct/inference" "$FEATURES_FILE" "$MODEL_NAME" \
        "$RESULTS_DIR/direct/sae_features"

    echo "=== [$TARGET_GENDER] Auditor: SAE feature descriptions ==="
    ./user_gender/scripts/run_auditor_sae_descriptions.sh \
        "$RESULTS_DIR/direct/sae_features_filtered" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/direct/audit/sae_feature_descriptions"

    echo "=== [$TARGET_GENDER] Auditor: SAE tokens ==="
    ./user_gender/scripts/run_auditor_sae_tokens.sh \
        "$RESULTS_DIR/direct/sae_features_filtered" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/direct/audit/sae_tokens"
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (SAE direct, female + male) ==="
