#!/usr/bin/env python3
"""Requires: CPU. Per-prompt accuracy of the layer-11 readout: what the average hides.

Pronoun positions live in the prompt, not the response, so the 10 sampled responses per prompt aren't independent:
the true unit is the unique prompt. Groups the auditor's graded items (`success`) by (unique prompt, checkpoint),
computes each prompt's accuracy, and counts prompts that fail completely (0.0), are partial (0.1 to 0.8) or succeed
cleanly (0.9 to 1.0). Run: runs.LAYER11_READOUT[set]. Writes derived/user_gender/bimodal_breakdown_{set}.json;
plots/bimodal_breakdown_chart_combined.py draws the counts.
"""
import argparse

import runs
import stats
from meta import write_derived

GENDERS = ["female", "male"]


def categories(accs) -> dict:
    return {"fail": sum(1 for a in accs if a == 0.0), "partial": sum(1 for a in accs if 0.0 < a < 0.9),
            "succeed": sum(1 for a in accs if a >= 0.9), "n": len(accs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    args = parser.parse_args()
    run_id = runs.LAYER11_READOUT[args.prompt_set]
    per_prompt, counts = {}, {}
    for g in GENDERS:
        by_prompt = stats.prompt_success(runs.load(run_id, g))
        per_prompt[g] = [sum(v) / len(v) for _p, v in sorted(by_prompt.items())]
        counts[g] = categories(per_prompt[g])
        print(f"{g}: {counts[g]}")
    write_derived(f"bimodal_breakdown_{args.prompt_set}.json",
                  {"run": run_id, "per_prompt": per_prompt, "counts": counts}, __file__)


if __name__ == "__main__":
    main()
