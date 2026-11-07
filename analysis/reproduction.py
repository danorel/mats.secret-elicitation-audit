#!/usr/bin/env python3
"""Requires: CPU. Reproduction numbers: the paper's five white-box methods and the layer-11 readout, one prompt set.

Reads cached audit grades for every method (runs.toml ids), both checkpoints, and writes the two-checkpoint average
with its 95% interval from the paired prompt-level cluster bootstrap (stats.cluster_ci) to
derived/user_gender/reproduction_bar_chart_{set}.json, one row per method in METRICS.md's order with the layer-11
readout under Acts Tokens at layer 23. figures: plots/reproduction_bar_chart.py, plots/layer_shift_chart.py.

The layer-11 row is runs.LAYER11_READOUT[set]: the bf16 Acts Tokens run on direct (68.05%), the float32
gender-direction run on standard (56.80%, the only layer-11 run there).
"""
import argparse

import runs
import stats
from meta import write_derived

# key (plotstyle.toml's methods entities) -> run id without the prompt set
METHODS = {
    "io": "io",
    "logit_lens": "logit_lens.l23",
    "residual_tokens": "acts_tokens.l23",
    "acts_tokens_l11": None,  # runs.LAYER11_READOUT
    "sae_feature_descriptions": "sae_desc.l23",
    "sae_tokens": "sae_tokens.l23",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    args = parser.parse_args()

    run_ids = {k: (f"{r}.{args.prompt_set}" if r else runs.LAYER11_READOUT[args.prompt_set]) for k, r in METHODS.items()}
    metrics = {k: (runs.load(rid, "female"), runs.load(rid, "male")) for k, rid in run_ids.items()}
    ci = stats.cluster_ci({k: {"by_f": stats.prompt_success(f), "by_m": stats.prompt_success(m)}
                           for k, (f, m) in metrics.items()})
    rows = []
    for k, (f, m) in metrics.items():
        f_acc, m_acc = f["metrics"]["mean_accuracy"] * 100, m["metrics"]["mean_accuracy"] * 100
        _point, lo, hi = ci[k]
        rows.append({"key": k, "run": run_ids[k], "female": f_acc, "male": m_acc, "avg": (f_acc + m_acc) / 2,
                     "ci95_lo": lo, "ci95_hi": hi})
        print(f"{k:26s} female={f_acc:6.2f}  male={m_acc:6.2f}  avg={(f_acc + m_acc) / 2:6.2f}  95% CI=[{lo:6.2f}, {hi:6.2f}]")
    write_derived(f"reproduction_bar_chart_{args.prompt_set}.json", rows, __file__)


if __name__ == "__main__":
    main()
