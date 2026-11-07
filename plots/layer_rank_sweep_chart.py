#!/usr/bin/env python3
"""Median full-vocabulary rank of the best-matching gender word vs. layer, both checkpoints: the 30-item pilot.

Draws derived/user_gender/layer_rank_sweep_chart.json (written by analysis/layer_rank_sweep.py) as a styles-plot
sweep. Figure: figures/post/layer_rank_sweep.png.
"""
import argparse

GENDERS = ["female", "male"]


def draw(data, out_dir):
    """Rows = the two checkpoints; median rank per layer on a log axis with its p25-p75 band, a dashed rule and a
    point at layer 11 (the layer picked from this pilot, so the registry marks the point post hoc)."""
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    lines, points = [], []
    for gender in GENDERS:
        layers = [d["layer"] for d in data[gender]]
        lines.append(sp.Line(gender, "method", layers, [d["median"] for d in data[gender]], cell=f"line.{gender}",
                             band=([d["p25"] for d in data[gender]], [d["p75"] for d in data[gender]]),
                             interval="iqr", median=True))
        points.append(sp.Point(gender, "method", 11, cell=f"l11.{gender}", key=f"pilot_rank.l11.{gender}",
                               label=True))
    # better (a lower rank) reads upward: the log axis is drawn inverted; ranks are capped at 5001 (README caption)
    lo = min(d["p25"] for g in GENDERS for d in data[g])
    hi = max(d["p75"] for g in GENDERS for d in data[g])
    sweep = sp.Sweep(GENDERS, "layer", "rank", lines, kind="checkpoints", points=points,
                     rules=[sp.Rule(11, "layer 11"), sp.Rule(23, "layer 23 (published)")],
                     xticks=[0, 5, 10, 15, 20, 24], ylim=(hi * 1.15, lo / 1.15),
                     yticks=[t for t in (30, 100, 300, 1000, 3000) if lo / 1.15 <= t <= hi * 1.15])
    return save(sp.Figure("layer_rank_sweep", v).add(sweep), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived("layer_rank_sweep_chart.json"), args.out_dir)


if __name__ == "__main__":
    main()
