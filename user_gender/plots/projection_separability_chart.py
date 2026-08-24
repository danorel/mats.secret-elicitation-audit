#!/usr/bin/env python3
"""Histogram of the gender_direction `mean_projection` field, split by true target gender.

Reads apply_gender_direction.py's output JSONs directly (no model, no GPU needed) for a
given layer and set of variants (contrastive / random_control / base_model_control), and
bins each variant's `mean_projection` values by true target gender. This is the same
distribution shown in the "Signal vs. Collapse" artifact from session 3 - that HTML had the
binned data hand-embedded; this script regenerates it from the actual result files so it's
reproducible for a different layer or after a new run, instead of a one-off snapshot.

Writes a JSON shaped for direct use by an HTML/JS chart (edges + per-variant/per-gender
counts), and optionally a static PNG via matplotlib for a quick look without an artifact.
"""
import argparse
import glob
import json
import os

DEFAULT_VARIANTS = ["contrastive", "random_control", "base_model_control"]
GENDERS = ["female", "male"]


def load_projections(results_root: str, gender: str, layer: int, variant: str):
    pattern = os.path.join(
        results_root, f"results_{gender}", "direct", "gender_direction_tokens",
        f"layer{layer}_{variant}", "*.json",
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No result file matching {pattern}")
    if len(files) > 1:
        print(f"Warning: {len(files)} files match {pattern}, using {files[0]}")
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
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
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--output_json", type=str, default=None,
                         help="Default: plots/output/gender_direction_hist_layer{layer}.json")
    parser.add_argument("--output_png", type=str, default=None,
                         help="If set, also render a static matplotlib PNG small-multiples figure.")
    args = parser.parse_args()

    raw = {}
    all_values = []
    for variant in args.variants:
        raw[variant] = {}
        for gender in GENDERS:
            vals = load_projections(args.results_root, gender, args.layer, variant)
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

    output_json = args.output_json or os.path.join("plots", "output", f"gender_direction_hist_layer{args.layer}.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved histogram data to {output_json}")

    if args.output_png:
        plot_png(out, args.layer, args.output_png)
        print(f"Saved figure to {args.output_png}")


def plot_png(hist_data, layer, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = hist_data["edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    variants = list(hist_data["variants"].keys())

    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.4), sharey=True)
    if len(variants) == 1:
        axes = [axes]

    colors = {"female": "#2a78d6", "male": "#eb6834"}
    for ax, variant in zip(axes, variants):
        for gender in GENDERS:
            d = hist_data["variants"][variant][gender]
            frac = [c / d["n"] for c in d["counts"]]
            ax.step(centers, frac, where="mid", color=colors[gender], label=gender, linewidth=2)
            ax.fill_between(centers, frac, step="mid", color=colors[gender], alpha=0.12)
        ax.axvline(0, color="0.6", linewidth=1, linestyle="-")
        ax.set_title(variant, fontsize=10)
        ax.set_xlabel("projection onto gender_direction")
    axes[0].set_ylabel("share of rollouts")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"gender_direction projection · layer {layer} · direct set", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=150)


if __name__ == "__main__":
    main()
