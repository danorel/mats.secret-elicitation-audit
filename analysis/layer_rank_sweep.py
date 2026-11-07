#!/usr/bin/env python3
"""Requires: CPU. Median full-vocabulary rank of the best-matching gender word vs. layer, both checkpoints (the pilot).

Reads the 30-item pilot (10 unique prompts x 3 response variants; run id rank_pilot.l0to24.direct) and writes the
per-layer median/p25/p75/min/max to derived/user_gender/layer_rank_sweep_chart.json;
plots/layer_rank_sweep_chart.py draws it. For the full-scale (n=990), documented-word-list version see
gender_word_rank_sweep.py. p25/p75 are numpy's linear-interpolated percentiles (before 7 Nov 2026 they were
vals[n // 4] and vals[3n // 4]). The pilot's ranks are capped at 5001: a word outside the top 5000 is recorded as 5001.
"""
import argparse
import statistics

import numpy as np

import runs
from meta import write_derived

GENDERS = ["female", "male"]


def summarize(ranks_by_layer: dict):
    layers = sorted((int(l) for l in ranks_by_layer), key=int)
    out = []
    for layer in layers:
        vals = sorted(ranks_by_layer[str(layer)])
        out.append({
            "layer": layer,
            "median": statistics.median(vals),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
            "min": vals[0],
            "max": vals[-1],
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    data = {g: summarize(runs.load("rank_pilot.l0to24.direct", g, kind="ranks")) for g in GENDERS}
    write_derived("layer_rank_sweep_chart.json", data, __file__)


if __name__ == "__main__":
    main()
