#!/usr/bin/env python3
"""Figure: sign-separability vs. decoded auditor accuracy, direct vs. standard set, layer 11.

Draws derived/user_gender/separability_vs_accuracy.json (written by analysis/separability_vs_accuracy.py, which says
where each number comes from) as a styles-plot dot table: rows = prompt sets, one column per quantity, the gap printed
as a number. Figure: figures/post/separability_vs_accuracy.png.
"""
import argparse

PROMPT_SETS = ["direct", "standard"]


def draw(separability, decoded, gap, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()

    def est(d, name, unit_key):
        return {p: sp.Est(d[p], cell=f"{name}.{p}", key=f"{unit_key}.{p}", label=True) for p in PROMPT_SETS}

    table = sp.DotTable(PROMPT_SETS, [
        sp.DotColumn("separability", "separability", (40, 104), (50, 75, 100),
                     series=[sp.Series("readout", est(separability, "sep", "separability.layer11"))],
                     rules=[sp.Rule(50, "chance")]),
        sp.DotColumn("auditor accuracy", "accuracy", (40, 104), (50, 75, 100),
                     series=[sp.Series("method", est(decoded, "acc", "decoded_accuracy.layer11"))],
                     rules=[sp.Rule(50, "chance")]),
        sp.TextColumn("between-checkpoint gap", "projection_gap", est(gap, "gap", "projection_gap.layer11")),
    ], kind="prompt_sets")
    return save(sp.Figure("separability_vs_accuracy", v).add(table), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    d = derived("separability_vs_accuracy.json")
    draw(d["separability"], d["decoded_accuracy"], d["gap"], args.out_dir)


if __name__ == "__main__":
    main()
