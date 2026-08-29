#!/usr/bin/env python3
"""Figure: the two attribution controls, side by side - direction and position.

Left panel: cos(mean_delta_female - mean_delta_male, control_direction) for five
same-model, same-method control directions (formality/tense/topic/sentiment, built
identically to the real gender direction, plus a Gaussian random_control), against the
real gender direction. Source: semantic_null_cosine_control.py
(user_gender/plots/semantic_null_cosine_control_layer11_direct.json when regenerated -
needs a GPU to rebuild the base-model control directions and load the mean-delta vectors,
neither of which is present on this machine, so the values below are transcribed from the
verified run reported in FINDINGS.md's "What the direction did establish" section).

Right panel: between-checkpoint projection gap at three non-pronoun control positions
(BOS, last user-turn token, user-turn midpoint) vs. the pronoun-mean position, `standard`
set, layer 11. Source: non_pronoun_position_gap.py
(user_gender/plots/non_pronoun_position_gap_layer11_standard.json when regenerated - same
GPU caveat; values transcribed from FINDINGS.md's "The secret is stored as a constant"
section, position-control paragraph).

Usage:
    python3 user_gender/plots/attribution_controls_chart.py
"""
import argparse

# Left panel: cos(delta_diff, control/gender direction), layer 11, direct set.
DIRECTION_COSINES = {
    "formality": -0.0393,
    "tense": -0.0126,
    "topic": -0.0547,
    "sentiment": +0.0120,
    "random\n(Gaussian)": -0.0011,
    "gender\n(real)": +0.5335,
}

# Right panel: between-checkpoint projection gap, layer 11, standard set.
POSITION_GAPS = {
    "BOS": 0.90,
    "last user-\nturn token": 2.07,
    "user-turn\nmidpoint": 3.33,
    "pronoun-mean\n(reference)": 28.91,
}


def plot_png(output_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    flat_color = "#90a4ae"
    real_color = "#c2185b"

    # Left panel
    labels1 = list(DIRECTION_COSINES.keys())
    vals1 = list(DIRECTION_COSINES.values())
    colors1 = [real_color if "gender" in l else flat_color for l in labels1]
    bars1 = ax1.bar(labels1, vals1, color=colors1)
    ax1.axhline(0, color="0.3", linewidth=0.8)
    ax1.set_ylabel("cos(fine-tuning delta, direction)")
    ax1.set_title("Direction control\ncos with 5 same-model, same-method directions", fontsize=10)
    for b, v in zip(bars1, vals1):
        ax1.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 4 if v >= 0 else -14), textcoords="offset points",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax1.set_ylim(-0.12, 0.62)
    ax1.tick_params(axis="x", labelsize=8)

    # Right panel
    labels2 = list(POSITION_GAPS.keys())
    vals2 = list(POSITION_GAPS.values())
    colors2 = [real_color if "pronoun" in l else flat_color for l in labels2]
    bars2 = ax2.bar(labels2, vals2, color=colors2)
    ax2.set_ylabel("between-checkpoint projection gap")
    ax2.set_title("Position control\ngap at 3 non-pronoun positions vs. pronoun-mean", fontsize=10)
    for b, v in zip(bars2, vals2):
        ax2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("The two attribution controls", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_png, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_png", type=str, default="user_gender/plots/attribution_controls.png")
    args = parser.parse_args()
    plot_png(args.output_png)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
