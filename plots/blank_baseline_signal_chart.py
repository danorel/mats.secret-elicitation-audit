#!/usr/bin/env python3
"""Real signal chart: each method's accuracy minus its own blank-evidence baseline.

Draws derived/user_gender/blank_baseline_signal_chart.json (written by analysis/blank_baseline_signal.py): one row
per method in the vocabulary's order, the signal with ± SEM whiskers and printed values, a dashed rule at 0.
Figure: figures/post/blank_baseline_signal.png.
"""
import argparse

# the derived rows carry labels; their table rows in plotstyle.toml's signal_rows order, and which are this project's
ROW_OF_LABEL = {
    "Acts Desc": "acts_desc",
    "Acts Tokens": "acts_tokens",
    "Gender Direction (direct)": "gender_direction_direct",
    "Gender Direction (standard)": "gender_direction_standard",
    "SAE Desc": "sae_desc",
}


def draw(rows_by_key, out_dir):
    """One row per method (vocabulary order), the signal with ± SEM whiskers and printed values; the gender
    direction in its role colour, the other methods grey; a dashed rule at 0 (the blank-evidence baseline)."""
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    order = [e.key for e in v.entities("signal_rows")]

    def est(key):
        r = rows_by_key[key]
        return sp.Est(r["signal"], r["signal"] - r["sem"], r["signal"] + r["sem"], interval="sem",
                      cell=f"signal.{key}", key=f"signal.{key}", label=True)

    ours = [k for k in order if rows_by_key[k]["ours"]]
    col = sp.DotColumn(
        "accuracy over the blank-evidence baseline", "signal", (-5, 45), (0, 20, 40),
        series=[sp.Series("method", {k: est(k) for k in ours}),
                sp.Series("control", {k: est(k) for k in order if k not in ours}, control="the other methods")],
        rules=[sp.Rule(0, "blank-evidence baseline")],
    )
    fig = sp.Figure("blank_baseline_signal", v).add(sp.DotTable(order, [col], kind="signal_rows"))
    return save(fig, out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    rows = derived("blank_baseline_signal_chart.json")
    draw({ROW_OF_LABEL[r["label"]]: r for r in rows}, args.out_dir)


if __name__ == "__main__":
    main()
