#!/usr/bin/env python3
"""Requires: CPU. Per-item projections on the gender direction at one layer, both prompt sets, both variants.

The strip figures (plots/projection_*_chart.py) and the between-checkpoint gap
(separability_vs_accuracy.py) need every item's `mean_projection`, not a histogram. This copies them out of the
registered apply-stage runs (projections.l{layer}.{variant}.{prompt_set}) into
derived/user_gender/projection_values_layer{layer}.json: {prompt_set: {variant: {gender: [values]}}}.
"""
import argparse

from meta import write_derived
from projection_separability import GENDERS, load_projections

PROMPT_SETS = ["direct", "standard"]
VARIANTS = ["contrastive", "base_model_control"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=11)
    args = parser.parse_args()
    out = {ps: {v: {g: load_projections(g, ps, args.layer, v) for g in GENDERS} for v in VARIANTS}
           for ps in PROMPT_SETS}
    write_derived(f"projection_values_layer{args.layer}.json", out, __file__, indent=1)


if __name__ == "__main__":
    main()
