#!/usr/bin/env python3
"""Reproduction chart: the paper's five white-box methods and the layer-11 readout, one prompt set.

Draws derived/user_gender/reproduction_bar_chart_{prompt_set}.json (analysis/reproduction.py) as a styles-plot dot
table: methods in the vocabulary's fixed order with Acts Tokens at layer 11 directly under layer 23, 95% prompt-level
intervals, printed values, chance at 50. Figure: figures/{readme,writeup}/reproduction_{prompt_set}.png.
"""
import argparse


def draw(rows, prompt_set, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    by_key = {r["key"]: r for r in rows}
    order = [e.key for e in v.entities("methods") if e.key in by_key]

    def est(key):
        r = by_key[key]
        return sp.Est(r["avg"], r["ci95_lo"], r["ci95_hi"], interval="ci95",
                      cell=f"{prompt_set}.{key}", key=f"{prompt_set}.{key}.avg", label=True)

    base = [k for k in order if k != "acts_tokens_l11"]
    col = sp.DotColumn(
        "auditor accuracy", "accuracy", (40, 100), (50, 75, 100),
        series=[sp.Series("method", {"acts_tokens_l11": est("acts_tokens_l11")}, word="layer 11"),
                sp.Series("control", {k: est(k) for k in base}, control="the paper's baselines")],
        rules=[sp.Rule(50, "chance")],
    )
    return save(sp.Figure(f"reproduction_{prompt_set}", v).add(sp.DotTable(order, [col], kind="methods")), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived(f"reproduction_bar_chart_{args.prompt_set}.json"), args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
