#!/usr/bin/env python3
"""Is cos(mean_delta_female - mean_delta_male, gender_direction) = +0.5335 far from a
same-model, same-method baseline, or is residual-stream anisotropy inflating it?

FINDINGS.md ("What the direction did establish"): the fine-tuning delta's alignment with
the base model's contrastive gender direction is ~32 sigma against a *Gaussian* null, but
the residual stream is not isotropic, so two diff-of-means vectors built from the same
model on similar text may share far more cosine than a random Gaussian vector would - even
for an unrelated contrastive pair. The informative control is 3-5 contrastive directions
built the same way (base model, layer 11, diff-of-means over first-person-pronoun
positions), for features that have nothing to do with gender, scored against the same
fine-tuning delta difference.

This script:
  1. Builds control directions for 4 semantically-null axes (formality, tense, topic,
     sentiment - contrastive pairs in prompts/gender/control_contrastive_pairs_*.jsonl,
     each pair sharing the same first-person-pronoun-bearing sentence skeleton the real
     gender pairs use, so the extraction procedure is identical in every way except what
     varies between the two sides of each pair).
  2. Loads the fine-tuning delta difference vector (mean_delta_female - mean_delta_male at
     layer 11, `direct` set) saved by finetuning_delta_diagnostic.py.
  3. Reports cos(delta_diff, control_direction) for each axis, plus the real gender
     direction's 0.5335 and the existing Gaussian random_control direction, for reference.

Needs a GPU only for step 1 (base model, one load, ~35 pairs x 4 axes - a few minutes).
Step 2-3 are pure numpy. Run finetuning_delta_diagnostic.py --layer 11 --prompt_set direct
first if the mean_delta npy files under user_gender/plots/mean_delta_vectors/ don't exist.

Usage:
    python3 user_gender/plots/semantic_null_cosine_control.py --layer 11
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elicitation_methods.gender_direction import load_direction

CONTROL_AXES = ["formality", "tense", "topic", "sentiment"]
BASE_MODEL = "google/gemma-2-9b-it"


def build_control_direction(axis, layer, base_model):
    pairs_file = f"prompts/gender/control_contrastive_pairs_{axis}.jsonl"
    output_dir = f"user_gender/inputs/gender_direction/control_{axis}"
    npy_path = os.path.join(output_dir, "contrastive", f"direction_contrastive_layer{layer}.npy")
    if os.path.exists(npy_path):
        print(f"[{axis}] direction already built at {npy_path}, skipping build")
        return npy_path
    cmd = [
        sys.executable, "elicitation_methods/gender_direction.py", "--mode", "build",
        "--model_name", base_model, "--pairs_file", pairs_file,
        "--layers", str(layer), "--output_dir", output_dir,
    ]
    print(f"[{axis}] building control direction: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return npy_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--prompt_set", type=str, default="direct")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--gender_direction_file", type=str,
                         default="user_gender/inputs/gender_direction/contrastive/direction_contrastive_layer{layer}.npy")
    parser.add_argument("--random_control_file", type=str,
                         default="user_gender/inputs/gender_direction/random_control/direction_random_control_layer{layer}.npy")
    parser.add_argument("--mean_delta_dir", type=str, default="user_gender/plots/mean_delta_vectors")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    fm_path = os.path.join(args.mean_delta_dir, f"mean_delta_female_layer{args.layer}_{args.prompt_set}.npy")
    mm_path = os.path.join(args.mean_delta_dir, f"mean_delta_male_layer{args.layer}_{args.prompt_set}.npy")
    if not (os.path.exists(fm_path) and os.path.exists(mm_path)):
        print(f"ERROR: {fm_path} / {mm_path} not found. Run:\n"
              f"  python3 user_gender/plots/finetuning_delta_diagnostic.py --layer {args.layer} "
              f"--prompt_set {args.prompt_set}\nfirst (needs the updated script that saves mean_delta_*.npy).")
        sys.exit(1)
    fm = np.load(fm_path)
    mm = np.load(mm_path)
    delta_diff = fm - mm
    delta_diff_hat = delta_diff / np.linalg.norm(delta_diff)

    gender_dir = load_direction(args.gender_direction_file.format(layer=args.layer))
    cos_gender = float(np.dot(delta_diff_hat, gender_dir))

    random_dir = load_direction(args.random_control_file.format(layer=args.layer))
    cos_random = float(np.dot(delta_diff_hat, random_dir))

    control_cosines = {}
    for axis in CONTROL_AXES:
        npy_path = build_control_direction(axis, args.layer, args.base_model)
        d = load_direction(npy_path)
        control_cosines[axis] = float(np.dot(delta_diff_hat, d))

    vals = np.array(list(control_cosines.values()))
    n_dim = delta_diff.shape[0]
    gaussian_sd = 1.0 / np.sqrt(n_dim)

    print(f"\n=== cosine of (mean_delta_female - mean_delta_male) with each direction, layer {args.layer} ===")
    print(f"  gender direction (real)      : {cos_gender:+.4f}   <- the 0.5335 claim under test")
    print(f"  random_control (Gaussian)    : {cos_random:+.4f}   (theoretical sd={gaussian_sd:.4f} for n={n_dim})")
    for axis, c in control_cosines.items():
        print(f"  control: {axis:12s}       : {c:+.4f}")
    print(f"\n  semantic-null controls: mean={vals.mean():+.4f}  sd={vals.std(ddof=1):.4f}  "
          f"range=[{vals.min():+.4f}, {vals.max():+.4f}]")
    if vals.std(ddof=1) > 1e-9:
        z = (cos_gender - vals.mean()) / vals.std(ddof=1)
        print(f"  gender direction is {z:+.2f} empirical-null-sd from the semantic-null control mean")

    report = {
        "layer": args.layer, "prompt_set": args.prompt_set, "timestamp": datetime.now().isoformat(),
        "hidden_dim": int(n_dim), "gaussian_theoretical_sd": float(gaussian_sd),
        "cos_gender_direction": cos_gender, "cos_random_control_gaussian": cos_random,
        "cos_semantic_null_controls": control_cosines,
        "semantic_null_mean": float(vals.mean()), "semantic_null_sd": float(vals.std(ddof=1)),
        "semantic_null_min": float(vals.min()), "semantic_null_max": float(vals.max()),
    }
    out = args.output_json or os.path.join(
        "user_gender", "plots", f"semantic_null_cosine_control_layer{args.layer}_{args.prompt_set}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
