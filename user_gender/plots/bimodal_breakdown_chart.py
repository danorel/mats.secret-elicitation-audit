#!/usr/bin/env python3
"""Per-prompt accuracy breakdown for the layer-11 headline result - the bimodal finding.

The 68.55% average hides structure: pronoun positions live in the prompt, not the
response, so the 10 sampled responses per prompt aren't independent - the true unit is
the unique prompt. Groups guesses by (unique_prompt, checkpoint), computes each unit's
own accuracy, draws the distribution as a styles-plot histogram per checkpoint - female
at ceiling, male bimodal (most phrasings fail completely, a minority succeed completely),
not a smooth spread.

No GPU - reads the saved guessing JSON directly.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "analysis"))
import runs  # noqa: E402

GENDERS = ["female", "male"]


def load_guessing(gender: str, prompt_set: str):
    return runs.load(f"gender_direction.l11.{prompt_set}", gender, kind="guessing")["results"]


def per_prompt_accuracy(results, target_word: str):
    groups = defaultdict(list)
    for r in results:
        success = 1 if target_word in [w.lower() for w in r.get("guessed_words", [])] else 0
        groups[r["user_prompt"]].append(success)
    return {prompt: sum(vals) / len(vals) for prompt, vals in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--output_json", type=str, default=None,
                        help="Default: results/user_gender/bimodal_breakdown_{prompt_set}.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    args.output_json = args.output_json or f"results/user_gender/bimodal_breakdown_{args.prompt_set}.json"

    per_gender = {}
    for gender in GENDERS:
        results = load_guessing(gender, args.prompt_set)
        accs = per_prompt_accuracy(results, gender)
        per_gender[gender] = list(accs.values())
        print(f"{gender}: {len(accs)} unique prompts, "
              f"{sum(1 for a in accs.values() if a == 1.0)} at 100%, "
              f"{sum(1 for a in accs.values() if a == 0.0)} at 0%")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(per_gender, f, indent=2)
    print(f"Wrote {args.output_json}")

    draw(per_gender, args.prompt_set, args.out_dir)


BINS = [i / 10 for i in range(11)]


def draw(per_gender, prompt_set, out_dir):
    """Rows = the two checkpoints: a histogram of per-prompt accuracy in tenths."""
    import styles_plot as sp
    from figstyle import save, vocab

    d = sp.Distribution(GENDERS, "prompt_accuracy", per_gender, role="method", kind="checkpoints",
                        cells={g: f"dist.{g}" for g in GENDERS}, count_unit="prompts", bins=BINS, xlim=(0, 1),
                        xticks=[0, 0.5, 1])
    return save(sp.Figure(f"bimodal_breakdown_{prompt_set}", vocab()).add(d), out_dir)


if __name__ == "__main__":
    main()
