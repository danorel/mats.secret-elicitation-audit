#!/usr/bin/env python3
"""Headline figure: Acts Tokens read at layer 23 (published) vs layer 11, direct set, against the best references.

Draws derived/user_gender/reproduction_bar_chart_direct.json (analysis/reproduction.py: both Acts Tokens rows with
their 95% prompt-level intervals, and SAE Desc, the best white-box method, reproduced) and
derived/user_gender/published_metrics.json (analysis/published_metrics.py: prefill, the best black-box method, as
published in the paper and not rerun here) as one dot-table row joined by a grey connector, with chance at 50.
Figure: figures/post/layer_shift_direct.png.
"""
import argparse


def draw(rows, published, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    by_key = {r["key"]: r for r in rows}
    prefill = published["user_gender"]["Direct Secret-keeping model"]["I/O + Prefill"]["mean"]

    def est(key, cell):
        r = by_key[key]
        return sp.Est(r["avg"], r["ci95_lo"], r["ci95_hi"], interval="ci95", cell=cell,
                      key=f"direct.{key}.avg", label=True)

    col = sp.DotColumn(
        "auditor accuracy", "accuracy", (40, 100), (50, 75, 100),
        series=[sp.Series("control", {"acts_tokens": est("residual_tokens", "l23")}, control="the published layer (23)"),
                sp.Series("method", {"acts_tokens": est("acts_tokens_l11", "l11")}, word="layer 11")],
        connect=("control", "method"),
        rules=[sp.Rule(50, "chance"),
               sp.Rule(by_key["sae_feature_descriptions"]["avg"], "best white-box, SAE Desc (reproduced): {}",
                       unit="accuracy", key="direct.sae_feature_descriptions.avg"),
               sp.Rule(prefill, "best black-box, prefill (published, not rerun): {}", unit="accuracy",
                       key="published.direct.prefill")],
    )
    return save(sp.Figure("layer_shift_direct", v).add(sp.DotTable(["acts_tokens"], [col], kind="headline")), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived("reproduction_bar_chart_direct.json"), derived("published_metrics.json"), args.out_dir)


if __name__ == "__main__":
    main()
