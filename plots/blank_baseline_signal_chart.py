#!/usr/bin/env python3
"""Real signal chart: each method's accuracy minus its own blank-evidence baseline, direct set.

Draws derived/user_gender/blank_baseline_signal_chart.json (analysis/blank_baseline_signal.py): one row per method in
the vocabulary's order, the signal with its 95% prompt-level interval and printed value, a dashed rule at 0.
Figure: figures/post/blank_baseline_signal.png.
"""
import argparse


def draw(rows, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    by_key = {r["key"]: r for r in rows}
    order = [e.key for e in v.entities("signal_rows")]

    def est(key):
        r = by_key[key]
        return sp.Est(r["signal"], r["ci95_lo"], r["ci95_hi"], interval="ci95",
                      cell=f"signal.{key}", key=f"signal.{key}", label=True)

    col = sp.DotColumn(
        "accuracy over the blank-evidence baseline", "signal", (-5, 45), (0, 20, 40),
        series=[sp.Series("method", {"acts_tokens_l11": est("acts_tokens_l11")}, word="layer 11"),
                sp.Series("control", {k: est(k) for k in order if k != "acts_tokens_l11"}, control="the other methods")],
        rules=[sp.Rule(0, "blank-evidence baseline")],
    )
    return save(sp.Figure("blank_baseline_signal", v).add(sp.DotTable(order, [col], kind="signal_rows")), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived("blank_baseline_signal_chart.json"), args.out_dir)


if __name__ == "__main__":
    main()
