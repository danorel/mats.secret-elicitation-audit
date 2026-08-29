#!/usr/bin/env python3
"""Figure 6: Gender Direction projection at layer 11, contrastive variant, direct vs.
standard set, side by side on a shared y-axis.

No GPU: reads gender_direction.py (--mode apply) output JSONs directly via
projection_separability_chart.load_projections, same data source as
projection_separability_layer11_{direct,standard}.png - this figure isolates just the
`contrastive` panel from each (dropping `base_model_control`, which is Figure 5's story,
not this one) and puts them on one shared axis so the near-identical spread is visually
comparable rather than split across two separately-scaled files.

Usage:
    python3 user_gender/plots/figure6_projection_direct_vs_standard.py
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from projection_separability_chart import GENDERS, load_projections  # noqa: E402


def plot_png(results_root, layer, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"female": "#c2185b", "male": "#1565c0"}
    prompt_sets = ["direct", "standard"]
    titles = {"direct": "direct\n(secret is behaviorally relevant)",
              "standard": "standard\n(ordinary chat, gender never comes up)"}

    data = {}
    for prompt_set in prompt_sets:
        data[prompt_set] = {g: load_projections(results_root, g, prompt_set, layer, "contrastive")
                             for g in GENDERS}

    all_vals = [v for ps in data.values() for g in ps.values() for v in g]
    ymin, ymax = min(all_vals), max(all_vals)
    pad = 0.06 * (ymax - ymin)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for ax, prompt_set in zip(axes, prompt_sets):
        pos, vals, face_colors = [], [], []
        for i, gender in enumerate(GENDERS):
            vals.append(data[prompt_set][gender])
            pos.append(i)
            face_colors.append(colors[gender])
        parts = ax.violinplot(vals, positions=pos, showmedians=True, widths=0.8)
        for body, color in zip(parts["bodies"], face_colors):
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor(color)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("0.3")
            parts[key].set_linewidth(1)
        ax.axhline(0, color="0.6", linewidth=1, linestyle="--", zorder=0)
        ax.set_xticks(pos)
        ax.set_xticklabels(GENDERS)
        ax.set_title(titles[prompt_set], fontsize=10)
        n_f, n_m = len(data[prompt_set]["female"]), len(data[prompt_set]["male"])
        ax.text(0.5, 0.02, f"n={n_f + n_m}", transform=ax.transAxes, ha="center", fontsize=8, color="0.4")

    axes[0].set_ylabel("projection onto gender_direction")
    axes[0].set_ylim(ymin - pad, ymax + pad)
    fig.suptitle("Same axis, same gap, whether or not gender is the topic\n"
                 "layer 11, contrastive direction", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_png, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--output_png", type=str,
                         default="user_gender/plots/figure6_projection_direct_vs_standard.png")
    args = parser.parse_args()
    plot_png(args.results_root, args.layer, args.output_png)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
