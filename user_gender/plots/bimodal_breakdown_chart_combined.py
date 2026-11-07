#!/usr/bin/env python3
"""Per-prompt auditor accuracy, direct and standard set side by side, per checkpoint.

No GPU: reads the already-saved per-prompt accuracy arrays
(bimodal_breakdown_direct.json, bimodal_breakdown_standard.json - both produced by
bimodal_breakdown_chart.py --prompt_set {direct,standard}). A styles-plot histogram grid:
rows = the two checkpoints, columns = the two prompt sets, one shared count axis.

Usage:
    python3 user_gender/plots/bimodal_breakdown_chart_combined.py
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bimodal_breakdown_chart import BINS, GENDERS  # noqa: E402

COLUMNS = {"direct": "direct set", "standard": "standard set"}


def draw(data, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    values = {(g, COLUMNS[p]): data[p][g] for p in COLUMNS for g in GENDERS}
    cells = {(g, COLUMNS[p]): f"dist.{g}.{p}" for p in COLUMNS for g in GENDERS}
    d = sp.Distribution(GENDERS, "prompt_accuracy", values, role="method", kind="checkpoints",
                        columns=list(COLUMNS.values()), cells=cells, count_unit="prompts", bins=BINS, xlim=(0, 1),
                        xticks=[0, 0.5, 1])
    return save(sp.Figure("bimodal_breakdown_combined", vocab()).add(d), out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direct_json", type=str, default="results/user_gender/bimodal_breakdown_direct.json")
    parser.add_argument("--standard_json", type=str, default="results/user_gender/bimodal_breakdown_standard.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    data = {}
    for prompt_set, path in (("direct", args.direct_json), ("standard", args.standard_json)):
        with open(path, "r", encoding="utf-8") as f:
            data[prompt_set] = json.load(f)
    draw(data, args.out_dir)


if __name__ == "__main__":
    main()
