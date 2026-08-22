#!/bin/bash

# SAE Feature Descriptions / SAE Tokens for the `standard` prompt set
# (`direct` already done in 02_run_sae_direct.sh). Completes baseline
# reproduction symmetry with the published table (I/O, LL Tokens, Acts
# Tokens already have both standard+direct; SAE Desc/Tokens only had
# direct). Lower priority than `direct` itself - see
# log/2026-08-22-session-02.md - but cheap to fill in given the batch-size
# fix from that session (commit 6aec84d) already applied.
#
# NEEDS GPU.

set -e

cd "$(dirname "$0")/../../"

AUDITOR_MODEL="google/gemma-3-4b-it"
FEATURES_FILE="sae_features_l23.jsonl"

run_pair() {
    local MODEL_NAME="$1"
    local TARGET_GENDER="$2"
    local RESULTS_DIR="user_gender/results_${TARGET_GENDER}"

    echo "=== [$TARGET_GENDER] SAE feature extraction + filtering (standard) ==="
    ./user_gender/scripts/run_get_sae_features.sh \
        "$RESULTS_DIR/standard/inference" "$FEATURES_FILE" "$MODEL_NAME" \
        "$RESULTS_DIR/standard/sae_features"

    echo "=== [$TARGET_GENDER] Auditor: SAE feature descriptions (standard) ==="
    ./user_gender/scripts/run_auditor_sae_descriptions.sh \
        "$RESULTS_DIR/standard/sae_features_filtered" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/standard/audit/sae_feature_descriptions"

    echo "=== [$TARGET_GENDER] Auditor: SAE tokens (standard) ==="
    ./user_gender/scripts/run_auditor_sae_tokens.sh \
        "$RESULTS_DIR/standard/sae_features_filtered" "$AUDITOR_MODEL" "$TARGET_GENDER" \
        "$RESULTS_DIR/standard/audit/sae_tokens"
}

run_pair "bcywinski/gemma-2-9b-it-user-female" "female"
run_pair "bcywinski/gemma-2-9b-it-user-male" "male"

echo "=== ALL DONE (SAE standard, female + male) ==="
