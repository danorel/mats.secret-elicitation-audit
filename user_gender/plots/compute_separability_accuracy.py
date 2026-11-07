#!/usr/bin/env python3
"""Sign-threshold classification accuracy for gender_direction's `mean_projection`,
computed from already-cached apply-stage JSONs (no GPU, no forward pass).

Same sign convention as `projection_separability_sweep.py` (direction = mean(acts_female)
- mean(acts_male), so female is "correct" at projection >= 0): this script answers the
same question from cached per-item results instead of a fresh forward pass, which is only
valid when a full-scale (`candidate_size: null`) apply-stage run already exists for the
requested (layer, prompt_set, variant).

Exists to check whether the near-perfect direct-set separability reported by
`projection_separability_sweep.py` (session 4) also holds on the `standard` set. If it
does, separability tracks checkpoint identity rather than secret-relevant content, since
target gender and checkpoint identity are the same variable in this organism, and the
`standard` set's topics are unrelated to gender.

Usage:
    python3 user_gender/plots/compute_separability_accuracy.py --layer 11 --prompt_set standard
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from projection_separability_chart import GENDERS, load_projections  # noqa: E402

EXPECTED_SIGN = {"female": 1, "male": -1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--variant", type=str, default="contrastive",
                         choices=["contrastive", "random_control", "base_model_control"])
    parser.add_argument("--prompt_set", type=str, default="standard", choices=["direct", "standard"])
    parser.add_argument("--output_json", type=str, default=None,
                         help="Default: results/user_gender/separability_accuracy_layer{layer}_{prompt_set}.json")
    args = parser.parse_args()

    correct_by_gender, n_by_gender = {}, {}
    for gender in GENDERS:
        values = load_projections(gender, args.prompt_set, args.layer, args.variant)
        correct = sum(1 for v in values if (1 if v >= 0 else -1) == EXPECTED_SIGN[gender])
        correct_by_gender[gender] = correct
        n_by_gender[gender] = len(values)
        print(f"{gender:6s} n={len(values):4d} correct={correct:4d} acc={100 * correct / len(values):6.2f}%")

    total_correct = sum(correct_by_gender.values())
    total_n = sum(n_by_gender.values())
    accuracy = total_correct / total_n
    print(f"TOTAL layer={args.layer} prompt_set={args.prompt_set} variant={args.variant}: "
          f"n={total_n} acc={accuracy * 100:.2f}%")

    output_json = args.output_json or os.path.join(
        "results", "user_gender", f"separability_accuracy_layer{args.layer}_{args.prompt_set}.json")
    out = {
        "layer": args.layer,
        "prompt_set": args.prompt_set,
        "variant": args.variant,
        "correct_by_gender": correct_by_gender,
        "n_by_gender": n_by_gender,
        "accuracy": accuracy,
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
