#!/usr/bin/env python3
"""Figure: sign-separability vs. decoded auditor accuracy, direct vs. standard set, layer 11.

No GPU: reads already-computed accuracy numbers. Sign-separability values come from
projection_separability_sweep.json (direct, layer 11) and a `--variant contrastive` rerun of
compute_separability_accuracy.py on the standard set (compute_separability_accuracy.py's
cached default output for standard is the base_model_control variant, not contrastive - see
separability_accuracy_layer11_standard_contrastive.json). Decoded accuracy is the Gender
Direction / Acts Tokens layer-11 auditor result from METRICS.md, reproduced in FINDINGS.md's
"The secret is stored as a constant" section. The between-checkpoint projection offset
(29.03 direct, 28.83 standard) is a different unit (raw projection, not a percentage) and is
annotated under each x-position rather than plotted as a third line.

Slopegraph form: separability stays flat near ceiling while decoded accuracy drops - two
lines whose shape *is* the claim, rather than four bars whose heights have to be compared by
eye. Both plotted series share one axis (0-100%) against the same chance line (50%), which a
bar chart alternative could also honor if the slopegraph reads as too unfamiliar.

Usage:
    python3 user_gender/plots/separability_vs_accuracy_chart.py
"""
import argparse

SEPARABILITY = {"direct": 98.99, "standard": 100.00}
DECODED_ACCURACY = {"direct": 68.05, "standard": 56.80}
OFFSET = {"direct": 29.03, "standard": 28.83}


def plot_png(output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prompt_sets = ["direct", "standard"]
    x = [0, 1]
    sep_vals = [SEPARABILITY[p] for p in prompt_sets]
    acc_vals = [DECODED_ACCURACY[p] for p in prompt_sets]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(x, sep_vals, marker="o", color="#1565c0", linewidth=2.5, markersize=9,
            label="sign-separability", zorder=3)
    ax.plot(x, acc_vals, marker="o", color="#c2185b", linewidth=2.5, markersize=9,
            label="decoded auditor accuracy", zorder=3)

    for xi, v in zip(x, sep_vals):
        ax.annotate(f"{v:.2f}%", (xi, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#1565c0")
    for xi, v in zip(x, acc_vals):
        ax.annotate(f"{v:.2f}%", (xi, v), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color="#c2185b")

    ax.axhline(50, color="0.55", linestyle="--", linewidth=1, zorder=0)
    ax.text(1.06, 50, "chance", fontsize=8, color="0.4", va="center")

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(40, 108)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"direct\n(secret is behaviorally relevant)\noffset = {OFFSET['direct']:.2f}",
        f"standard\n(ordinary chat, gender never comes up)\noffset = {OFFSET['standard']:.2f}",
    ], fontsize=9)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Separability stays flat while decoded accuracy drops\nlayer 11, both sets",
                 fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_png", type=str,
                         default="user_gender/plots/separability_vs_accuracy.png")
    args = parser.parse_args()
    plot_png(args.output_png)
    print(f"Wrote {args.output_png}")
    for p in ("direct", "standard"):
        print(f"{p:10s} separability={SEPARABILITY[p]:6.2f}%  decoded_accuracy={DECODED_ACCURACY[p]:6.2f}%  "
              f"offset={OFFSET[p]:.2f}")


if __name__ == "__main__":
    main()
