#!/usr/bin/env python3
"""Per-prompt accuracy breakdown for the layer-11 headline result - the bimodal finding.

The 68.55% average hides structure: pronoun positions live in the prompt, not the
response, so the 10 sampled responses per prompt aren't independent - the true unit is
the unique prompt. Groups guesses by (unique_prompt, checkpoint), computes each unit's
own accuracy, plots the distribution - female at ceiling, male bimodal (most phrasings
fail completely, a minority succeed completely), not a smooth spread.

No GPU - reads the saved guessing JSON directly.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

GENDERS = ["female", "male"]


def load_guessing(results_root: str, gender: str):
    d = os.path.join(results_root, "results", gender, "direct", "final", "gender_direction",
                      "audit", "layer11_finetuned_model")
    files = glob.glob(os.path.join(d, "guessing_*.json"))
    if not files:
        raise FileNotFoundError(f"No guessing file under {d}")
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)["results"]


def per_prompt_accuracy(results, target_word: str):
    groups = defaultdict(list)
    for r in results:
        success = 1 if target_word in [w.lower() for w in r.get("guessed_words", [])] else 0
        groups[r["user_prompt"]].append(success)
    return {prompt: sum(vals) / len(vals) for prompt, vals in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--output_json", type=str, default="user_gender/plots/bimodal_breakdown.json")
    parser.add_argument("--output_png", type=str, default="user_gender/plots/bimodal_breakdown_chart.png")
    args = parser.parse_args()

    per_gender = {}
    for gender in GENDERS:
        results = load_guessing(args.results_root, gender)
        accs = per_prompt_accuracy(results, gender)
        per_gender[gender] = list(accs.values())
        print(f"{gender}: {len(accs)} unique prompts, "
              f"{sum(1 for a in accs.values() if a == 1.0)} at 100%, "
              f"{sum(1 for a in accs.values() if a == 0.0)} at 0%")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(per_gender, f, indent=2)
    print(f"Wrote {args.output_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"female": "#c2185b", "male": "#1565c0"}
    bins = [i / 10 for i in range(11)]
    for ax, gender in zip(axes, GENDERS):
        ax.hist(per_gender[gender], bins=bins, color=colors[gender], edgecolor="white")
        ax.set_title(f"{gender} (n={len(per_gender[gender])} unique prompts)")
        ax.set_xlabel("per-prompt auditor accuracy")
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_ylabel("number of unique prompts")
    fig.suptitle("Layer 11 headline result, broken down by unique prompt - not a smooth average", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
