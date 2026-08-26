#!/bin/bash

# Runs, in order, the two experiments identified in log/2026-08-26-session-06.md's "Next
# session" list as blocking the write-up's headline attribution claim (R1: does the
# accuracy gain come from the contrastive direction, or from which layer/position it
# reads?) and its layer-11 separability claim (does 98.99%/100% separability reflect real
# secret-relevant signal, or checkpoint identity?).
#
# Step 1 costs no GPU time: it recomputes sign-threshold separability at layer 11 from
# apply-stage results already cached under user_gender/results/ (session 4's full-scale
# runs). Step 2 needs the GPU: it runs the paper's own Acts Tokens method at layer 11
# instead of its published layer 23, decode and auditor grading included (~30 min on an
# A100 for both checkpoints).
#
# Usage:
#   ./user_gender/scripts/run_verification_experiments.sh

set -e

cd "$(dirname "$0")/../../"
source .venv/bin/activate

echo "############################################################"
echo "# Step 1/2: layer-11 sign-separability, direct + standard  #"
echo "# (no GPU - reads cached apply-stage results)               #"
echo "############################################################"
echo ""
echo "--- direct set (sanity check against session 4's 98.99% full-sweep number) ---"
python3 user_gender/plots/compute_separability_accuracy.py --layer 11 --prompt_set direct
echo ""
echo "--- standard set (the decisive check: does separability hold where content is unrelated to gender?) ---"
python3 user_gender/plots/compute_separability_accuracy.py --layer 11 --prompt_set standard

echo ""
echo "############################################################"
echo "# Step 2/2: Acts Tokens @ layer 11 (paper's method,        #"
echo "# non-published layer), direct set - needs GPU             #"
echo "############################################################"
echo ""
./user_gender/scripts/run_acts_tokens_layer11_check.sh

echo ""
echo "############################################################"
echo "# Done. Compare against these already-established numbers: #"
echo "#   Acts Tokens @ layer 23 (published):     60.25%          #"
echo "#   Gender Direction @ layer 11 (headline):  68.55%         #"
echo "#   Layer-11 separability, direct (session 4): 98.99%       #"
echo "############################################################"
