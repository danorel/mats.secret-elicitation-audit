#!/usr/bin/env python3
"""Requires: CPU. Reproduction numbers: the paper's five white-box methods and the layer-11 readout, one prompt set.

Reads cached audit grades for every method (runs.toml ids), both checkpoints; writes the two-checkpoint average and an
item-level bootstrap SEM per method (stats.bootstrap_sem) to derived/user_gender/reproduction_bar_chart_{set}.json,
rows sorted by average. plots/reproduction_bar_chart.py draws it. The visual form of METRICS.md's
Secret-keeping model table.

The SEM treats the 10 responses per prompt as independent draws and so understates the uncertainty; see
EXPERIMENTS.md #resampling.
"""
import argparse

import runs
import stats
from meta import write_derived

# key -> (run id without the prompt set, label); the layer-11 row is gender_direction.l11
BASELINE_METHODS = {
    "io": ("io", "I/O (no evidence)"),
    "logit_lens": ("logit_lens.l23", "Logit Lens Tokens"),
    "residual_tokens": ("acts_tokens.l23", "Acts Tokens"),
    "sae_tokens": ("sae_tokens.l23", "SAE Tokens"),
    "sae_feature_descriptions": ("sae_desc.l23", "SAE Desc"),
}


def row(run_id: str, label: str, ours: bool) -> dict:
    f_metrics, m_metrics = runs.load(run_id, "female"), runs.load(run_id, "male")
    f_acc = f_metrics["metrics"]["mean_accuracy"] * 100
    m_acc = m_metrics["metrics"]["mean_accuracy"] * 100
    return {"label": label, "female": f_acc, "male": m_acc, "avg": (f_acc + m_acc) / 2,
            "sem": stats.bootstrap_sem(f_metrics, m_metrics), "ours": ours}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    args = parser.parse_args()

    rows = []
    for run, label in BASELINE_METHODS.values():
        try:
            rows.append(row(f"{run}.{args.prompt_set}", label, ours=False))
        except (KeyError, FileNotFoundError):
            print(f"skipping {label}: no {args.prompt_set}-set data")
    rows.append(row(f"gender_direction.l11.{args.prompt_set}", "Gender Direction (L11)", ours=True))

    rows.sort(key=lambda r: r["avg"])
    for r in rows:
        print(f"{r['label']:22s} female={r['female']:6.2f}  male={r['male']:6.2f}  avg={r['avg']:6.2f} +/- {r['sem']:.2f}")
    write_derived(f"reproduction_bar_chart_{args.prompt_set}.json", rows, __file__)


if __name__ == "__main__":
    main()
