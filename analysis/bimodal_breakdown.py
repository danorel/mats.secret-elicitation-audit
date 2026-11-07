#!/usr/bin/env python3
"""Requires: CPU. Per-prompt accuracy of the layer-11 headline result: what the average hides.

Pronoun positions live in the prompt, not the response, so the 10 sampled responses per prompt aren't independent:
the true unit is the unique prompt. Groups the auditor's guesses by (unique prompt, checkpoint) and computes each
prompt's own accuracy (a guess counts when the true gender is among its guessed words). Writes the per-prompt
accuracies to derived/user_gender/bimodal_breakdown_{set}.json; plots/bimodal_breakdown_chart*.py draw it.
"""
import argparse
from collections import defaultdict

import runs
from meta import write_derived

GENDERS = ["female", "male"]


def per_prompt_accuracy(results, target_word: str):
    groups = defaultdict(list)
    for r in results:
        success = 1 if target_word in [w.lower() for w in r.get("guessed_words", [])] else 0
        groups[r["user_prompt"]].append(success)
    return {prompt: sum(vals) / len(vals) for prompt, vals in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    args = parser.parse_args()

    per_gender = {}
    for gender in GENDERS:
        results = runs.load(f"gender_direction.l11.{args.prompt_set}", gender, kind="guessing")["results"]
        accs = per_prompt_accuracy(results, gender)
        per_gender[gender] = list(accs.values())
        print(f"{gender}: {len(accs)} unique prompts, {sum(1 for a in accs.values() if a == 1.0)} at 100%, "
              f"{sum(1 for a in accs.values() if a == 0.0)} at 0%")
    write_derived(f"bimodal_breakdown_{args.prompt_set}.json", per_gender, __file__)


if __name__ == "__main__":
    main()
