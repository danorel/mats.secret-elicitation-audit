#!/usr/bin/env python3
"""Requires: CPU. Distribution of the gender direction's `mean_projection`, by true target gender, one layer.

Reads the apply-stage results registered in runs.toml (projections.l{layer}.{variant}.{prompt_set}) and bins each
variant's per-item `mean_projection` by true gender: the raw signal a downstream decode has to work with. Writes
histogram-ready JSON (edges + per-variant/per-gender counts) to
derived/user_gender/projection_separability_layer{layer}_{prompt_set}[_contrastive_only].json. The figures draw the
per-item values instead (projection_values.py).
"""
import argparse

import runs
from meta import write_derived

DEFAULT_VARIANTS = ["contrastive", "base_model_control"]
GENDERS = ["female", "male"]


def load_projections(gender: str, prompt_set: str, layer: int, variant: str):
    """Per-item mean_projection of the apply-stage run registered as projections.l{layer}.{variant}.{prompt_set}
    (runs.toml names the exact file; it used to be found by searching experiments/ and taking the file with the most
    processed pairs, with ties broken by filesystem order)."""
    data = runs.load(f"projections.l{layer}.{variant}.{prompt_set}", gender, kind="projections")
    return [r["mean_projection"] for r in data["results"] if "error" not in r]


def build_histogram(values, edges):
    n_bins = len(edges) - 1
    vmin, vmax = edges[0], edges[-1]
    bin_width = (vmax - vmin) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = int((v - vmin) / bin_width)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1
    return counts


def output_name(layer: int, prompt_set: str, variants) -> str:
    suffix = "_contrastive_only" if list(variants) == ["contrastive"] else ""
    return f"projection_separability_layer{layer}_{prompt_set}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--n_bins", type=int, default=32)
    args = parser.parse_args()

    raw, all_values = {}, []
    for variant in args.variants:
        raw[variant] = {}
        for gender in GENDERS:
            vals = load_projections(gender, args.prompt_set, args.layer, variant)
            raw[variant][gender] = vals
            all_values.extend(vals)
            print(f"{variant:20s} {gender:6s} n={len(vals):4d} mean={sum(vals)/len(vals):8.2f}")

    vmin, vmax = min(all_values), max(all_values)
    edges = [round(vmin + i * (vmax - vmin) / args.n_bins, 3) for i in range(args.n_bins + 1)]
    out = {"edges": edges, "variants": {}}
    for variant in args.variants:
        out["variants"][variant] = {g: {"counts": build_histogram(raw[variant][g], edges), "n": len(raw[variant][g])}
                                    for g in GENDERS}
    write_derived(output_name(args.layer, args.prompt_set, args.variants), out, __file__)


if __name__ == "__main__":
    main()
