#!/usr/bin/env python3
"""Requires: CPU. Median full-vocabulary rank of the best-matching gender word vs. layer, both checkpoints (the pilot).

Reads the 30-item pilot (10 unique prompts x 3 response variants; run id rank_pilot.l0to24.direct) and writes the
per-layer median/p25/p75/min/max to derived/user_gender/layer_rank_sweep_chart.json;
plots/layer_rank_sweep_chart.py draws it. For the full-scale (n=990), documented-word-list version see
gender_word_rank_sweep.py. p25/p75 are vals[n // 4] and vals[3n // 4], not interpolated percentiles; kept as computed
so the JSON stays comparable with earlier runs.
"""
import argparse
import statistics

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
            "p25": vals[len(vals) // 4],
            "p75": vals[(3 * len(vals)) // 4],
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
