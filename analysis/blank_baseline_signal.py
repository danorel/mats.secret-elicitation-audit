#!/usr/bin/env python3
"""Requires: CPU. Real signal: each method's accuracy minus its own blank-evidence baseline, direct set.

The grader has a content-independent default guess whose strength depends on the evidence field's shape, not its
content (README.md Finding 4). This scores each method as distance from a matched blank-evidence control - same
template, same pipeline, evidence content replaced with a constant placeholder. Writes
derived/user_gender/blank_baseline_signal_chart.json; plots/blank_baseline_signal_chart.py draws it.

Interval: 95% from the paired prompt-level cluster bootstrap. The real and the blank run are graded on the same
prompts, so each draw resamples prompts once and takes real - blank. The layer-11 row is runs.LAYER11_READOUT["direct"]
(the bf16 run, 68.05%). The standard-set layer-11 signal (+6.05) is only in the README's table: its blank baseline is
the direct one, so it is not a like-for-like row here.
"""
import argparse

import runs
import stats
from meta import write_derived

# key (plotstyle.toml's signal_rows entities) -> (real run id, blank run id), all on the direct set
METHODS = {
    "acts_desc": ("acts_desc.direct", "acts_desc.blank.direct"),
    "acts_tokens": ("acts_tokens.l23.direct", "blank.token_list.direct"),
    "acts_tokens_l11": (runs.LAYER11_READOUT["direct"], "blank.token_list.direct"),
    "sae_desc": ("sae_desc.l23.direct", "blank.sae_desc.direct"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    rows = []
    for key, (real_id, blank_id) in METHODS.items():
        data = {name: {"by_f": stats.prompt_success(runs.load(rid, "female")),
                       "by_m": stats.prompt_success(runs.load(rid, "male"))}
                for name, rid in (("real", real_id), ("blank", blank_id))}
        draws = stats.cluster_draws(data, 20000, 1)
        real_point, real_boot = stats.cluster_accuracy(data["real"], *draws)
        blank_point, blank_boot = stats.cluster_accuracy(data["blank"], *draws)
        lo, hi = stats.ci95(real_boot - blank_boot)
        rows.append({"key": key, "real_run": real_id, "blank_run": blank_id, "real_accuracy": float(real_point),
                     "blank_baseline": float(blank_point), "signal": float(real_point - blank_point),
                     "ci95_lo": lo, "ci95_hi": hi})
        print(f"{key:16s} real={real_point:6.2f}  blank={blank_point:6.2f}  signal={real_point - blank_point:+6.2f}  "
              f"95% CI=[{lo:+6.2f}, {hi:+6.2f}]")
    write_derived("blank_baseline_signal_chart.json", rows, __file__)


if __name__ == "__main__":
    main()
