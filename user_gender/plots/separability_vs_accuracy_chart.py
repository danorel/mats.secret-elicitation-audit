#!/usr/bin/env python3
"""Figure: sign-separability vs. decoded auditor accuracy, direct vs. standard set, layer 11.

No GPU: every number is read from saved run files (it used to be typed into this script):

- sign-separability, direct: results/user_gender/projection_separability_sweep.json, accuracy_by_layer["11"] (contrastive direction,
  n=1980). separability_accuracy_layer11_direct.json is the base_model_control variant, so it is not used.
- sign-separability, standard: separability_accuracy_layer11_standard_contrastive.json (a `--variant contrastive`
  rerun of compute_separability_accuracy.py; its cached default output for standard is base_model_control).
- decoded auditor accuracy: the mean of the two checkpoints' mean_accuracy. direct reads Acts Tokens at layer 11
  (results/*/direct/final/acts_tokens_layer11/audit, 68.05%, the dtype-matched readout); standard reads the gender
  direction's layer-11 audit (results/*/standard/final/gender_direction/audit/layer11_finetuned_model, 56.80%), the
  only layer-11 readout graded on the standard set. These are the numbers in README.md's Finding 3
  table.
- between-checkpoint gap: mean projection (female) - mean projection (male), contrastive direction, from the
  gender_direction.py --mode apply outputs (projection_separability_chart.load_projections).

Drawn as a styles-plot dot table: rows = prompt sets, one column per quantity, the gap printed as a number.

Usage:
    python3 user_gender/plots/separability_vs_accuracy_chart.py
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from projection_separability_chart import GENDERS, load_projections  # noqa: E402
import runs  # noqa: E402  (analysis/, on sys.path via projection_separability_chart)

PROMPT_SETS = ["direct", "standard"]
DECODED_RUNS = {  # run ids in runs.toml
    "direct": "acts_tokens.l11.direct",
    "standard": "gender_direction.l11.standard",
}


def load_numbers(data_dir: str, layer: int):
    with open(os.path.join(data_dir, "projection_separability_sweep.json"), "r", encoding="utf-8") as f:
        sweep = json.load(f)
    with open(os.path.join(data_dir, f"separability_accuracy_layer{layer}_standard_contrastive.json"), "r",
              encoding="utf-8") as f:
        standard = json.load(f)
    if standard["variant"] != "contrastive" or standard["layer"] != layer:
        raise ValueError(f"expected the contrastive layer-{layer} rerun, got {standard['variant']} "
                         f"layer {standard['layer']}")
    separability = {"direct": sweep["accuracy_by_layer"][str(layer)] * 100, "standard": standard["accuracy"] * 100}

    decoded = {}
    for prompt_set, run_id in DECODED_RUNS.items():
        accs = [runs.load(run_id, g)["metrics"]["mean_accuracy"] for g in GENDERS]
        decoded[prompt_set] = sum(accs) / len(accs) * 100

    gap = {}
    for prompt_set in PROMPT_SETS:
        f_vals, m_vals = (load_projections(g, prompt_set, layer, "contrastive") for g in GENDERS)
        gap[prompt_set] = sum(f_vals) / len(f_vals) - sum(m_vals) / len(m_vals)
    return separability, decoded, gap


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="results/user_gender",
                        help="Where the separability JSONs live (projection_separability_sweep.json, ...)")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()

    separability, decoded, gap = load_numbers(args.data_dir, args.layer)
    for p in PROMPT_SETS:
        print(f"{p:10s} separability={separability[p]:6.2f}%  decoded_accuracy={decoded[p]:6.2f}%  "
              f"offset={gap[p]:.2f}")
    draw(separability, decoded, gap, args.out_dir)


if __name__ == "__main__":
    main()
