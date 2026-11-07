#!/usr/bin/env python3
"""Figure: sign-threshold separability of the gender-direction projection, per layer (no GPU).

Reads projection_separability_sweep.py's JSON (projection_separability_sweep.json for the direct set,
projection_separability_sweep_{prompt_set}.json otherwise) and draws a styles-plot sweep: separability per layer,
a dashed rule and a point at layer 11 (the layer the auditor pipeline uses), and a point at the best layer of this
sweep (picked from the plotted data, so the registry marks it post hoc).

Usage:
    python3 user_gender/plots/projection_separability_sweep_chart.py [--prompt_set direct]
"""
import argparse
import json
import os

RESULTS_DIR = os.path.join("results", "user_gender")


def load_sweep(prompt_set: str):
    suffix = "" if prompt_set == "direct" else f"_{prompt_set}"
    with open(os.path.join(RESULTS_DIR, f"projection_separability_sweep{suffix}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    layers = [int(l) for l in data["layers"]]
    return layers, [data["accuracy_by_layer"][str(l)] * 100 for l in layers]


def draw(layers, accuracy, prompt_set, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    # the first layer with the highest separability, as the old chart's max() picked it
    best = layers[max(range(len(layers)), key=lambda i: accuracy[i])]
    line = sp.Line(prompt_set, "readout", layers, accuracy, cell=f"line.{prompt_set}")
    # no printed values: layer 11's label collides with the neighbouring best-layer dot (SP601), and the best
    # layer's lands a layer away from its dot; both values are in the README's Proxy 2 table
    points = [sp.Point(prompt_set, "readout", 11, cell=f"l11.{prompt_set}")]
    if best != 11:
        points.append(sp.Point(prompt_set, "readout", best, cell=f"best.{prompt_set}"))
    sweep = sp.Sweep([prompt_set], "layer", "separability", [line], kind="prompt_sets", header="separability",
                     points=points, rules=[sp.Rule(11, "layer 11")], hrules=[sp.Rule(50, "chance")], ylim=(40, 102), yticks=[50, 75, 100],
                     xticks=[10, 15, 20, 25])
    return save(sp.Figure(f"projection_separability_sweep_{prompt_set}", v).add(sweep), out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    layers, accuracy = load_sweep(args.prompt_set)
    draw(layers, accuracy, args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
