#!/usr/bin/env python3
"""Requires: CPU. Real signal: each method's accuracy minus its own blank-evidence baseline.

The grader has a content-independent default guess whose strength depends on the evidence field's shape, not its
content (README.md Finding 4). Raw accuracy therefore mixes real informativeness with that default. This scores each
method as distance from a matched blank-evidence control - same template, same pipeline, evidence content replaced
with a constant placeholder. Writes derived/user_gender/blank_baseline_signal_chart.json, rows sorted by signal;
plots/blank_baseline_signal_chart.py draws it.

Error bars: item-level bootstrap SEM over the real-accuracy metric only (stats.bootstrap_sem). The blank-evidence
baseline is treated as fixed (n=1000, 500/gender), so this understates the uncertainty of the difference somewhat.
"""
import argparse

import runs
import stats
from meta import write_derived

# run ids without the prompt set (runs.toml names the files); `real` is read on the method's prompt set, `blank` on
# --blank_prompt_set
METHODS = {
    "acts_desc": {
        "label": "Acts Desc",
        "real": "acts_desc",
        "blank": "acts_desc.blank",
        "prompt_set": "direct",
    },
    "acts_tokens": {
        "label": "Acts Tokens",
        "real": "acts_tokens.l23",
        "blank": "blank.token_list",
        "prompt_set": "direct",
    },
    "gender_direction_direct": {
        "label": "Gender Direction (direct)",
        "real": "gender_direction.l11",
        "blank": "blank.token_list",
        "prompt_set": "direct",
        "ours": True,
    },
    "gender_direction_standard": {
        "label": "Gender Direction (standard)",
        "real": "gender_direction.l11",
        "blank": "blank.token_list",
        "prompt_set": "standard",
        "ours": True,
    },
    "sae_desc": {
        "label": "SAE Desc",
        "real": "sae_desc.l23",
        "blank": "blank.sae_desc",
        "prompt_set": "direct",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # blank-evidence controls were only run on the direct prompt set; standard-set methods are scored against the
    # same blank baseline (the grader's default is about the evidence field's shape, not the prompt set).
    parser.add_argument("--blank_prompt_set", type=str, default="direct")
    args = parser.parse_args()

    rows = []
    for spec in METHODS.values():
        f_real = runs.load(f"{spec['real']}.{spec['prompt_set']}", "female")
        m_real = runs.load(f"{spec['real']}.{spec['prompt_set']}", "male")
        f_blank = runs.load(f"{spec['blank']}.{args.blank_prompt_set}", "female")
        m_blank = runs.load(f"{spec['blank']}.{args.blank_prompt_set}", "male")
        real_acc = (f_real["metrics"]["mean_accuracy"] + m_real["metrics"]["mean_accuracy"]) / 2 * 100
        blank_acc = (f_blank["metrics"]["mean_accuracy"] + m_blank["metrics"]["mean_accuracy"]) / 2 * 100
        rows.append({"label": spec["label"], "real_accuracy": real_acc, "blank_baseline": blank_acc,
                     "signal": real_acc - blank_acc, "sem": stats.bootstrap_sem(f_real, m_real),
                     "ours": spec.get("ours", False)})

    rows.sort(key=lambda r: r["signal"])
    for r in rows:
        print(f"{r['label']:28s} real={r['real_accuracy']:6.2f}  blank={r['blank_baseline']:6.2f}  "
              f"signal={r['signal']:+6.2f} +/- {r['sem']:.2f}")
    write_derived("blank_baseline_signal_chart.json", rows, __file__)


if __name__ == "__main__":
    main()
