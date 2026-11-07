#!/usr/bin/env python3
"""Reproduction chart: the paper's five white-box methods and the layer-11 readout, one prompt set.

Draws derived/user_gender/reproduction_bar_chart_{prompt_set}.json (written by analysis/reproduction.py) as a
styles-plot dot table: methods in the vocabulary's fixed order, ± SEM whiskers, printed values, chance at 50.
Figure: figures/post/reproduction_{prompt_set}.png.
"""
import argparse

# the derived rows carry labels; their table rows in plotstyle.toml's methods order
ROW_OF_LABEL = {
    "I/O (no evidence)": "io",
    "Logit Lens Tokens": "logit_lens",
    "Acts Tokens": "residual_tokens",
    "SAE Tokens": "sae_tokens",
    "SAE Desc": "sae_feature_descriptions",
    "Gender Direction (L11)": "gender_direction",
}


def draw(rows_by_key, prompt_set, out_dir):
    """Methods as rows (vocabulary order), one accuracy column: dots with ± SEM whiskers and printed values, the
    gender direction in its role colour, the paper's baselines grey, a chance rule at 50."""
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    order = [e.key for e in v.entities("methods") if e.key in rows_by_key]

    def est(key):
        r = rows_by_key[key]
        return sp.Est(r["avg"], r["avg"] - r["sem"], r["avg"] + r["sem"], interval="sem",
                      cell=f"{prompt_set}.{key}", key=f"{prompt_set}.{key}.avg", label=True)

    base = [k for k in order if k != "gender_direction"]
    col = sp.DotColumn(
        "auditor accuracy", "accuracy", (40, 100), (50, 75, 100),
        series=[sp.Series("method", {"gender_direction": est("gender_direction")}),
                sp.Series("control", {k: est(k) for k in base}, control="the paper's baselines")],
        rules=[sp.Rule(50, "chance")],
    )
    fig = sp.Figure(f"reproduction_{prompt_set}", v).add(sp.DotTable(order, [col], kind="methods"))
    return save(fig, out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    rows = derived(f"reproduction_bar_chart_{args.prompt_set}.json")
    draw({ROW_OF_LABEL[r["label"]]: r for r in rows}, args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
