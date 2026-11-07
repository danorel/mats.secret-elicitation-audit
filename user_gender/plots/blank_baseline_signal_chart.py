#!/usr/bin/env python3
"""Real signal chart: each method's accuracy minus its own blank-evidence baseline.

The grader has a content-independent default guess whose strength depends on the
evidence field's shape, not its content (see FINDINGS.md's grader-bias section).
Raw accuracy therefore mixes real informativeness with that default. This plots
each method's accuracy as distance from a matched blank-evidence control - same
template, same pipeline, evidence content replaced with a constant placeholder -
which is the honest per-method signal.

Error bars: raw-item bootstrap over the real-accuracy metric only (combined
female+male per-item `success` flags), same convention as reproduction_bar_chart.py.
The blank-evidence baseline itself is treated as fixed: it's estimated on n=1000
(500/gender) same as the real metric, and is the more stable of the two numbers in
practice (see FINDINGS.md), so this understates true uncertainty in the difference
somewhat - a paired bootstrap over both would be tighter.
"""
import argparse
import glob
import json
import os

import numpy as np

METHODS = {
    "acts_desc": {
        "label": "Acts Desc",
        "real": ("final", "acts_desc", "audit"),
        "blank": ("final", "acts_desc", "null_control_audit"),
        "prompt_set": "direct",
    },
    "acts_tokens": {
        "label": "Acts Tokens",
        "real": ("baseline", "audit", "residual_tokens"),
        "blank": ("final", "auditor_calibration", "acts_tokens_template", "audit"),
        "prompt_set": "direct",
    },
    "gender_direction_direct": {
        "label": "Gender Direction (direct)",
        "real": ("final", "gender_direction", "audit", "layer11_finetuned_model"),
        "blank": ("final", "auditor_calibration", "acts_tokens_template", "audit"),
        "prompt_set": "direct",
        "ours": True,
    },
    "gender_direction_standard": {
        "label": "Gender Direction (standard)",
        "real": ("final", "gender_direction", "audit", "layer11_finetuned_model"),
        "blank": ("final", "auditor_calibration", "acts_tokens_template", "audit"),
        "prompt_set": "standard",
        "ours": True,
    },
    "sae_desc": {
        "label": "SAE Desc",
        "real": ("baseline", "audit", "sae_feature_descriptions"),
        "blank": ("final", "auditor_calibration", "sae_desc_template", "audit"),
        "prompt_set": "direct",
    },
}


def load_metrics(results_root: str, gender: str, prompt_set: str, path_parts):
    d = os.path.join(results_root, "results", gender, prompt_set, *path_parts)
    files = glob.glob(os.path.join(d, "metrics.json")) or glob.glob(os.path.join(d, "metrics_*.json"))
    if not files:
        raise FileNotFoundError(f"No metrics file under {d}")
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_sem(f_metrics, m_metrics, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    f_succ = np.array([r["success"] for r in f_metrics["results"]])
    m_succ = np.array([r["success"] for r in m_metrics["results"]])
    combined = np.concatenate([f_succ, m_succ])
    n = len(combined)
    boot_means = np.array([rng.choice(combined, size=n, replace=True).mean() for _ in range(n_boot)])
    return boot_means.std() * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="user_gender")
    # blank-evidence controls were only run on the direct prompt set; standard-set
    # methods are scored against the same blank baseline (the grader's default is
    # about the evidence field's shape, not the prompt set).
    parser.add_argument("--blank_prompt_set", type=str, default="direct")
    parser.add_argument("--output_json", type=str, default="results/user_gender/blank_baseline_signal_chart.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()

    rows, rows_by_key = [], {}
    for key, spec in METHODS.items():
        f_real = load_metrics(args.results_root, "female", spec["prompt_set"], spec["real"])
        m_real = load_metrics(args.results_root, "male", spec["prompt_set"], spec["real"])
        f_blank = load_metrics(args.results_root, "female", args.blank_prompt_set, spec["blank"])
        m_blank = load_metrics(args.results_root, "male", args.blank_prompt_set, spec["blank"])

        real_acc = (f_real["metrics"]["mean_accuracy"] + m_real["metrics"]["mean_accuracy"]) / 2 * 100
        blank_acc = (f_blank["metrics"]["mean_accuracy"] + m_blank["metrics"]["mean_accuracy"]) / 2 * 100
        sem = bootstrap_sem(f_real, m_real)

        rows.append({
            "label": spec["label"],
            "real_accuracy": real_acc,
            "blank_baseline": blank_acc,
            "signal": real_acc - blank_acc,
            "sem": sem,
            "ours": spec.get("ours", False),
        })
        rows_by_key[key] = rows[-1]

    rows.sort(key=lambda r: r["signal"])
    for r in rows:
        print(f"{r['label']:28s} real={r['real_accuracy']:6.2f}  blank={r['blank_baseline']:6.2f}  "
              f"signal={r['signal']:+6.2f} +/- {r['sem']:.2f}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.output_json}")

    draw(rows_by_key, args.out_dir)


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

    ours = [k for k in order if METHODS[k].get("ours")]
    col = sp.DotColumn(
        "accuracy over the blank-evidence baseline", "signal", (-5, 45), (0, 20, 40),
        series=[sp.Series("method", {k: est(k) for k in ours}),
                sp.Series("control", {k: est(k) for k in order if k not in ours}, control="the other methods")],
        rules=[sp.Rule(0, "blank-evidence baseline")],
    )
    fig = sp.Figure("blank_baseline_signal", v).add(sp.DotTable(order, [col], kind="signal_rows"))
    return save(fig, out_dir)


if __name__ == "__main__":
    main()
