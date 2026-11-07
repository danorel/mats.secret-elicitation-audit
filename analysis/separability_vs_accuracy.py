#!/usr/bin/env python3
"""Requires: CPU. Sign-separability vs. decoded auditor accuracy vs. between-checkpoint gap, direct vs. standard, layer 11.

- sign-separability, direct: derived/user_gender/projection_separability_sweep.json, accuracy_by_layer["11"]
  (contrastive direction, n=1980). separability_accuracy_layer11_direct.json is the base_model_control variant, so it
  is not used.
- sign-separability, standard: derived/user_gender/separability_accuracy_layer11_standard_contrastive.json (a
  `--variant contrastive` rerun of compute_separability_accuracy.py).
- decoded auditor accuracy: the mean of the two checkpoints' mean_accuracy, from run acts_tokens.l11.direct (bf16,
  68.05%) and gender_direction.l11.standard (float32, 56.80%; the only layer-11 run on standard). These are the
  numbers in README.md's Finding 3 table.
- between-checkpoint gap: mean projection (female) - mean projection (male), contrastive direction, from the
  registered apply-stage runs.

Writes derived/user_gender/separability_vs_accuracy.json; plots/separability_vs_accuracy_chart.py draws it.
"""
import argparse

import runs
from meta import read_derived, write_derived
from projection_separability import GENDERS, load_projections

PROMPT_SETS = ["direct", "standard"]
DECODED_RUNS = {  # run ids in runs.toml
    "direct": "acts_tokens.l11.direct",
    "standard": "gender_direction.l11.standard",
}


def load_numbers(layer: int):
    sweep = read_derived("projection_separability_sweep.json")
    standard = read_derived(f"separability_accuracy_layer{layer}_standard_contrastive.json")
    if standard["variant"] != "contrastive" or standard["layer"] != layer:
        raise ValueError(f"expected the contrastive layer-{layer} rerun, got {standard['variant']} "
                         f"layer {standard['layer']}")
    separability = {"direct": sweep["accuracy_by_layer"][str(layer)] * 100, "standard": standard["accuracy"] * 100}

    decoded = {}
    for prompt_set, run_id in DECODED_RUNS.items():
        accs = [runs.load(run_id, g)["metrics"]["mean_accuracy"] for g in GENDERS]
        decoded[prompt_set] = sum(accs) / len(accs) * 100

    gap = {}
    for prompt_set in PROMPT_SETS:
        f_vals, m_vals = (load_projections(g, prompt_set, layer, "contrastive") for g in GENDERS)
        gap[prompt_set] = sum(f_vals) / len(f_vals) - sum(m_vals) / len(m_vals)
    return separability, decoded, gap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=11)
    args = parser.parse_args()
    separability, decoded, gap = load_numbers(args.layer)
    for p in PROMPT_SETS:
        print(f"{p:10s} separability={separability[p]:6.2f}%  decoded_accuracy={decoded[p]:6.2f}%  offset={gap[p]:.2f}")
    write_derived("separability_vs_accuracy.json",
                  {"layer": args.layer, "separability": separability, "decoded_accuracy": decoded, "gap": gap}, __file__)


if __name__ == "__main__":
    main()
