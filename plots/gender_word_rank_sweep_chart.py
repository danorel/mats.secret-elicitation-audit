#!/usr/bin/env python3
"""Figure: full-scale (n=990 per checkpoint) rank of the best gender word per layer (no GPU).

Reads gender_word_rank_sweep.py's JSON (gender_word_rank_sweep.json) and draws a styles-plot sweep: rows = the two
checkpoints, the median rank per layer on a log axis with its interquartile band (numpy's linear-interpolated 25th
and 75th percentiles), a dashed rule and a point at layer 11. The medians are README.md's "What we did"
table (Proxy 1). The 30-item pilot the layer was first picked from is layer_rank_sweep_chart.py.

Usage:
    python3 plots/gender_word_rank_sweep_chart.py
"""
import argparse
import statistics

import numpy as np

GENDERS = ["female", "male"]


def summarize(data: dict):
    layers = [int(l) for l in data["layers"]]
    out = {}
    for gender in GENDERS:
        by_layer = data["ranks_by_layer"][gender]
        out[gender] = {
            "layers": layers,
            "median": [statistics.median(by_layer[str(l)]) for l in layers],
            "p25": [float(np.percentile(by_layer[str(l)], 25)) for l in layers],
            "p75": [float(np.percentile(by_layer[str(l)], 75)) for l in layers],
        }
    return out


def draw(summary, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    lines = [sp.Line(g, "method", s["layers"], s["median"], cell=f"line.{g}", band=(s["p25"], s["p75"]),
                     interval="iqr", median=True) for g, s in summary.items()]
    points = [sp.Point(g, "method", 11, cell=f"l11.{g}", key=f"word_rank.l11.{g}", label=True) for g in summary]
    layers = summary[GENDERS[0]]["layers"]
    sweep = sp.Sweep(GENDERS, "layer", "rank", lines, kind="checkpoints", points=points,
                     rules=[sp.Rule(11, "layer 11")], xticks=[l for l in layers if l % 3 == 0])
    return save(sp.Figure("gender_word_rank_sweep", v).add(sweep), out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    from figstyle import derived

    summary = summarize(derived("gender_word_rank_sweep.json"))
    for g, s in summary.items():
        print(g, dict(zip(s["layers"], s["median"])))
    draw(summary, args.out_dir)


if __name__ == "__main__":
    main()
