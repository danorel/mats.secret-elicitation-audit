#!/usr/bin/env python3
"""Distribution of gender_direction's `mean_projection`, by true target gender, one layer.

No GPU: reads gender_direction.py's (--mode apply) output JSONs directly and bins each
variant's (contrastive / random_control / base_model_control) `mean_projection` by true
gender - the raw signal a downstream decode has to work with.

Writes histogram-ready JSON (edges + per-variant/per-gender counts) and optionally a
static PNG (violin plots).
"""
import argparse
import glob
import json
import os

DEFAULT_VARIANTS = ["contrastive", "random_control", "base_model_control"]
GENDERS = ["female", "male"]


def infer_variant(metadata: dict) -> str:
    if "google/gemma-2-9b-it" in metadata.get("model_name", "") and "user-" not in metadata.get("model_name", ""):
        return "base_model_control"
    if "random_control" in metadata.get("direction_file", ""):
        return "random_control"
    return "contrastive"


def load_projections(results_root: str, gender: str, layer: int, variant: str):
    """Finds the apply-stage result file for (gender, layer, variant) by reading each
    candidate JSON's own metadata (target_layer, model_name, direction_file) rather than
    relying on folder-naming convention, which has varied across this project's
    experiments/ history. If multiple matches exist (e.g. an n=300 pilot and a later
    n=1000 full run), picks the one with the most processed pairs."""
    search_root = os.path.join(results_root, "results", gender, "direct", "experiments", "gender_direction")
    candidates = []
    for path in glob.glob(os.path.join(search_root, "**", "*.json"), recursive=True):
        if os.sep + "audit" + os.sep in path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data["metadata"]
        except (json.JSONDecodeError, KeyError):
            continue
        if meta.get("target_layer") != layer or infer_variant(meta) != variant:
            continue
        candidates.append((meta.get("processed_pairs", 0), path, data))

    if not candidates:
        raise FileNotFoundError(f"No apply-stage result found for gender={gender} layer={layer} variant={variant} "
                                 f"under {search_root}")
    candidates.sort(key=lambda c: c[0])
    _, best_path, data = candidates[-1]
    if len(candidates) > 1:
        print(f"Note: {len(candidates)} files matched gender={gender} layer={layer} variant={variant}, "
              f"using the one with the most processed pairs: {best_path}")
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
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--n_bins", type=int, default=32)
    parser.add_argument("--output_json", type=str, default=None,
                         help="Default: user_gender/plots/projection_separability_layer{layer}.json")
    parser.add_argument("--output_png", type=str, default=None,
                         help="If set, also render a static matplotlib PNG (violin plots).")
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

    output_json = args.output_json or os.path.join("user_gender", "plots", f"projection_separability_layer{args.layer}.json")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved histogram data to {output_json}")

    if args.output_png:
        plot_png(raw, args.layer, args.output_png)
        print(f"Saved figure to {args.output_png}")


def plot_png(raw, layer, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(raw.keys())
    colors = {"female": "#c2185b", "male": "#1565c0"}

    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.6), sharey=True)
    if len(variants) == 1:
        axes = [axes]

    for ax, variant in zip(axes, variants):
        positions, data, face_colors = [], [], []
        for i, gender in enumerate(GENDERS):
            data.append(raw[variant][gender])
            positions.append(i)
            face_colors.append(colors[gender])

        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
        for body, color in zip(parts["bodies"], face_colors):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("0.3")
            parts[key].set_linewidth(1)

        ax.axhline(0, color="0.6", linewidth=1, linestyle="--", zorder=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(GENDERS)
        ax.set_title(variant, fontsize=10)

    axes[0].set_ylabel("projection onto gender_direction")
    fig.suptitle(f"Gender Direction projection by true target gender · layer {layer} · direct set", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=150)


if __name__ == "__main__":
    main()
