#!/usr/bin/env python3
"""Gender Direction projection at layer 11, contrastive variant, direct vs.
standard set, side by side on a shared y-axis.

No GPU: reads gender_direction.py (--mode apply) output JSONs directly via
projection_separability_chart.load_projections, same data source as
projection_separability_layer11_{direct,standard}.png - this isolates just the
`contrastive` panel from each (dropping `base_model_control`, a separate story) and
puts them on one shared axis so the near-identical spread is visually comparable rather
than split across two separately-scaled files. Drawn as styles-plot strips
(figures/post/projection_direct_vs_standard.png).

Usage:
    python3 plots/projection_direct_vs_standard_chart.py
"""
import argparse

GENDERS = ["female", "male"]


COLUMNS = {"direct": "direct set", "standard": "standard set"}


def draw(values, out_dir):
    """Rows = the two checkpoints, columns = the two prompt sets, one shared x axis: each item a dot, the median a
    line (violins are not in the grammar)."""
    import styles_plot as sp
    from figstyle import save, vocab

    data, cells = {}, {}
    for prompt_set, col in COLUMNS.items():
        for g in GENDERS:
            data[(g, col)] = values[prompt_set]["contrastive"][g]
            cells[(g, col)] = f"dist.{g}.{prompt_set}"
    allv = [x for vals in data.values() for x in vals]
    pad = 0.05 * (max(allv) - min(allv))
    d = sp.Distribution(GENDERS, "projection", data, role="method", kind="checkpoints", style="strip", median=True,
                        columns=list(COLUMNS.values()), cells=cells, xlim=(min(allv) - pad, max(allv) + pad))
    return save(sp.Figure("projection_direct_vs_standard", vocab()).add(d), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived(f"projection_values_layer{args.layer}.json"), args.out_dir)


if __name__ == "__main__":
    main()
