#!/usr/bin/env python3
"""Median full-vocabulary rank of the best-matching gender word vs. layer, both genders.

Reads the real per-item rank data saved under
results/{female,male}/direct/experiments/gender_direction/layer_rank_sweep_0to24/ranks.json
(30 items/layer: 10 unique prompts x 3 response variants). This is the sweep that found
the broad 9-21 plateau (not a narrowing "sweet spot") and reopened layer 11 as the best
single point - see log/2026-08-24-session-04.md.

Writes a JSON shaped for direct use by an HTML/JS chart (per-layer median/p25/p75 per
gender), and a static PNG via matplotlib for a quick look.
"""
import argparse
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENDERS = ["female", "male"]


def load_ranks(gender: str):
    path = os.path.join(
        ROOT, "user_gender", "results", gender, "direct", "experiments",
        "gender_direction", "layer_rank_sweep_0to24", "ranks.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(ranks_by_layer: dict):
    layers = sorted((int(l) for l in ranks_by_layer), key=int)
    out = []
    for layer in layers:
        vals = sorted(ranks_by_layer[str(layer)])
        out.append({
            "layer": layer,
            "median": statistics.median(vals),
            "p25": vals[len(vals) // 4],
            "p75": vals[(3 * len(vals)) // 4],
            "min": vals[0],
            "max": vals[-1],
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", type=str,
                         default="user_gender/plots/layer_rank_sweep_chart.json")
    parser.add_argument("--output_png", type=str,
                         default="user_gender/plots/layer_rank_sweep_chart.png")
    args = parser.parse_args()

    data = {gender: summarize(load_ranks(gender)) for gender in GENDERS}

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output_json}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for gender, color in [("female", "#c2185b"), ("male", "#1565c0")]:
            layers = [d["layer"] for d in data[gender]]
            medians = [d["median"] for d in data[gender]]
            p25 = [d["p25"] for d in data[gender]]
            p75 = [d["p75"] for d in data[gender]]
            ax.plot(layers, medians, label=f"{gender} (median)", color=color, linewidth=2)
            ax.fill_between(layers, p25, p75, color=color, alpha=0.15)
        ax.axvline(11, color="gray", linestyle="--", linewidth=1, label="layer 11 (best single point)")
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_ylabel("full-vocabulary rank of best gender word (log scale, lower = easier to decode)")
        ax.set_title("Nearest-token rank of gender-related vocabulary vs. layer")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_png, dpi=150)
        print(f"Wrote {args.output_png}")
    except ImportError:
        print("matplotlib not available, skipping PNG")


if __name__ == "__main__":
    main()
