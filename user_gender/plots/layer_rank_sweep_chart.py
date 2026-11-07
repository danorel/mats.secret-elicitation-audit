#!/usr/bin/env python3
"""Median full-vocabulary rank of the best-matching gender word vs. layer, both genders.

Reads the 30-item pilot (10 unique prompts x 3 response variants) saved under
results/{female,male}/direct/experiments/gender_direction/layer_rank_sweep_0to24/ranks.json.
For the full-scale (n=1000), documented-word-list version, see gender_word_rank_sweep.py.

Writes per-layer median/p25/p75 JSON (for an HTML/JS chart) and a styles-plot sweep
(figures/post/layer_rank_sweep.png). p25/p75 are vals[n // 4] and vals[3n // 4], not
interpolated percentiles; kept as computed so the JSON stays comparable with earlier runs.
"""
import argparse
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GENDERS = ["female", "male"]


def load_ranks(gender: str):
    path = os.path.join(
        ROOT, "user_gender", "results", gender, "direct", "experiments",
        "gender_direction", "layer_rank_sweep_0to24", "ranks.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    sweep = sp.Sweep(GENDERS, "layer", "rank", lines, kind="checkpoints", points=points,
                     rules=[sp.Rule(11, "layer 11")], xticks=[0, 5, 10, 15, 20, 24])
    return save(sp.Figure("layer_rank_sweep", v).add(sweep), out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_json", type=str,
                         default="results/user_gender/layer_rank_sweep_chart.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()

    data = {gender: summarize(load_ranks(gender)) for gender in GENDERS}

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output_json}")

    draw(data, args.out_dir)


if __name__ == "__main__":
    main()
