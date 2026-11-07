#!/usr/bin/env python3
"""Distribution of gender_direction's `mean_projection`, by true target gender, one layer.

No GPU: reads gender_direction.py's (--mode apply) output JSONs directly and bins each
variant's (contrastive / random_control / base_model_control) `mean_projection` by true
gender - the raw signal a downstream decode has to work with.

Writes histogram-ready JSON (edges + per-variant/per-gender counts) and a styles-plot
strip figure (each item a dot, the median a line; violins are not in the grammar).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "analysis"))
import runs  # noqa: E402

DEFAULT_VARIANTS = ["contrastive", "random_control", "base_model_control"]
GENDERS = ["female", "male"]


def infer_variant(metadata: dict) -> str:
    if "google/gemma-2-9b-it" in metadata.get("model_name", "") and "user-" not in metadata.get("model_name", ""):
        return "base_model_control"
    if "random_control" in metadata.get("direction_file", ""):
        return "random_control"
    return "contrastive"


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


def main():
    parser = argparse.ArgumentParser(description="Bin gender_direction mean_projection by true target gender.")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--output_json", type=str, default=None,
                         help="Default: results/user_gender/projection_separability_layer{layer}.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()

    raw = {}
    all_values = []
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
        out["variants"][variant] = {}
        for gender in GENDERS:
            vals = raw[variant][gender]
            out["variants"][variant][gender] = {"counts": build_histogram(vals, edges), "n": len(vals)}

    output_json = args.output_json or os.path.join("results", "user_gender", f"projection_separability_layer{args.layer}.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved histogram data to {output_json}")

    draw(raw, args.layer, args.prompt_set, args.out_dir)


VARIANT_TERMS = {"contrastive": "contrastive", "random_control": "random control", "base_model_control": "base model"}
VARIANT_CONTROLS = {"random_control": "a random direction", "base_model_control": "the base model"}


def figure_id(layer: int, prompt_set: str, variants) -> str:
    """The ids keep the old PNG names: *_contrastive_only for the contrastive variant alone."""
    suffix = "_contrastive_only" if list(variants) == ["contrastive"] else ""
    return f"projection_separability_layer{layer}_{prompt_set}{suffix}"


def draw(raw, layer, prompt_set, out_dir):
    """One strip block per variant (each prompt a dot, a median line), rows = the two checkpoints, one shared
    x axis; the contrastive direction in its role colour, the control variants grey."""
    import styles_plot as sp
    from figstyle import save, vocab

    allv = [x for by_g in raw.values() for vals in by_g.values() for x in vals]
    lo, hi = min(allv), max(allv)
    pad = 0.05 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    fig = sp.Figure(figure_id(layer, prompt_set, raw), vocab())
    for variant, by_gender in raw.items():
        col = VARIANT_TERMS[variant]
        role = "method" if variant == "contrastive" else "control"
        fig.add(sp.Distribution(GENDERS, "projection", {(g, col): by_gender[g] for g in GENDERS}, role=role,
                                control=VARIANT_CONTROLS.get(variant), kind="checkpoints", style="strip", median=True,
                                columns=[col], xlim=xlim, panel_h=0.55,
                                cells={(g, col): f"dist.{variant}.{g}" for g in GENDERS}))
    return save(fig, out_dir)


if __name__ == "__main__":
    main()
