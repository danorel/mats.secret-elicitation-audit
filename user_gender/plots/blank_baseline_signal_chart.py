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
    parser.add_argument("--output_json", type=str, default="user_gender/plots/blank_baseline_signal_chart.json")
    parser.add_argument("--output_png", type=str, default="user_gender/plots/blank_baseline_signal_chart.png")
    args = parser.parse_args()

    rows = []
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

    rows.sort(key=lambda r: r["signal"])
    for r in rows:
        print(f"{r['label']:28s} real={r['real_accuracy']:6.2f}  blank={r['blank_baseline']:6.2f}  "
              f"signal={r['signal']:+6.2f} +/- {r['sem']:.2f}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.output_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in rows]
    signals = [r["signal"] for r in rows]
    sems = [r["sem"] for r in rows]
    colors = ["#c2185b" if r["ours"] else "#90a4ae" for r in rows]
    bars = ax.barh(labels, signals, xerr=sems, color=colors, capsize=4,
                    error_kw={"ecolor": "0.3", "elinewidth": 1.2})
    for bar, val, sem in zip(bars, signals, sems):
        ax.text(val + sem + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}pp", va="center", fontsize=9)
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1, label="blank-evidence baseline")
    ax.set_xlabel("accuracy minus own blank-evidence baseline (pp), ± bootstrap SEM")
    ax.set_xlim(-5, 45)
    ax.set_title("Real signal: distance from each method's own blank-evidence control")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
