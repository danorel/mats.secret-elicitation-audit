#!/usr/bin/env python3
"""Per-prompt auditor accuracy, direct (left pair) and standard (right pair), side by side.

No GPU: reads the already-saved per-prompt accuracy arrays
(bimodal_breakdown_direct.json, bimodal_breakdown_standard.json - both produced by
bimodal_breakdown_chart.py --prompt_set {direct,standard}). Four panels sharing one y-axis:
female-direct, male-direct, female-standard, male-standard.

Usage:
    python3 user_gender/plots/bimodal_breakdown_chart_combined.py
"""
import argparse
import json


def plot_png(direct_json, standard_json, output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(direct_json, "r", encoding="utf-8") as f:
        direct = json.load(f)
    with open(standard_json, "r", encoding="utf-8") as f:
        standard = json.load(f)

    colors = {"female": "#c2185b", "male": "#1565c0"}
    bins = [i / 10 for i in range(11)]
    panels = [("direct", "female"), ("direct", "male"), ("standard", "female"), ("standard", "male")]
    data = {"direct": direct, "standard": standard}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (prompt_set, gender) in zip(axes, panels):
        vals = data[prompt_set][gender]
        ax.hist(vals, bins=bins, color=colors[gender], edgecolor="white")
        n_zero = sum(1 for v in vals if v == 0.0)
        ax.set_title(f"{prompt_set} · {gender}\n(n={len(vals)}, {n_zero} at 0.0)", fontsize=10)
        ax.set_xlabel("per-prompt auditor accuracy")
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("number of unique prompts")

    for i in (1, 2):
        axes[i].spines["left"].set_alpha(0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct_json", type=str, default="user_gender/plots/bimodal_breakdown_direct.json")
    parser.add_argument("--standard_json", type=str, default="user_gender/plots/bimodal_breakdown_standard.json")
    parser.add_argument("--output_png", type=str,
                         default="user_gender/plots/bimodal_breakdown_chart_combined.png")
    args = parser.parse_args()
    plot_png(args.direct_json, args.standard_json, args.output_png)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
