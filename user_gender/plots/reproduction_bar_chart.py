#!/usr/bin/env python3
"""Reproduction bar chart: gender_direction vs. the paper's own baselines.

Reads audit metrics directly (no GPU) for every method's `direct`-set result, both
checkpoints, plots the balanced average as a sorted horizontal bar chart with error
bars - the visual form of METRICS.md's Direct Secret-keeping model table.

Error bars: raw-item bootstrap over combined female+male per-item `success` flags -
matches METRICS.md's own convention (closer to the paper's published stderr than
treating the 10 per-response-index accuracies as the sample). Likely understates true
uncertainty for gender_direction specifically, since its 10 responses per prompt aren't
independent draws - see FINDINGS.md.
"""
import argparse
import glob
import json
import os

import numpy as np

BASELINE_METHODS = {
    "io": ("baseline", "audit", "io", "I/O (no evidence)"),
    "logit_lens": ("baseline", "audit", "logit_lens", "Logit Lens Tokens"),
    "residual_tokens": ("baseline", "audit", "residual_tokens", "Acts Tokens"),
    "sae_tokens": ("baseline", "audit", "sae_tokens", "SAE Tokens"),
    "sae_feature_descriptions": ("baseline", "audit", "sae_feature_descriptions", "SAE Desc"),
}
GENDERS = ["female", "male"]


def load_metrics(results_root: str, gender: str, path_parts):
    d = os.path.join(results_root, "results", gender, "direct", *path_parts)
    files = glob.glob(os.path.join(d, "metrics_*.json"))
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
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--output_json", type=str, default="user_gender/plots/reproduction_bar_chart.json")
    parser.add_argument("--output_png", type=str, default="user_gender/plots/reproduction_bar_chart.png")
    args = parser.parse_args()

    rows = []
    for key, (a, b, c, label) in BASELINE_METHODS.items():
        f_metrics = load_metrics(args.results_root, "female", (a, b, c))
        m_metrics = load_metrics(args.results_root, "male", (a, b, c))
        f_acc = f_metrics["metrics"]["mean_accuracy"] * 100
        m_acc = m_metrics["metrics"]["mean_accuracy"] * 100
        sem = bootstrap_sem(f_metrics, m_metrics)
        rows.append({"label": label, "female": f_acc, "male": m_acc, "avg": (f_acc + m_acc) / 2,
                     "sem": sem, "ours": False})

    f_metrics = load_metrics(args.results_root, "female", ("final", "gender_direction", "audit", "layer11_finetuned_model"))
    m_metrics = load_metrics(args.results_root, "male", ("final", "gender_direction", "audit", "layer11_finetuned_model"))
    f_acc = f_metrics["metrics"]["mean_accuracy"] * 100
    m_acc = m_metrics["metrics"]["mean_accuracy"] * 100
    sem = bootstrap_sem(f_metrics, m_metrics)
    rows.append({"label": f"Gender Direction (L{args.layer})", "female": f_acc, "male": m_acc,
                 "avg": (f_acc + m_acc) / 2, "sem": sem, "ours": True})

    rows.sort(key=lambda r: r["avg"])
    for r in rows:
        print(f"{r['label']:22s} female={r['female']:6.2f}  male={r['male']:6.2f}  "
              f"avg={r['avg']:6.2f} +/- {r['sem']:.2f}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {args.output_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in rows]
    avgs = [r["avg"] for r in rows]
    sems = [r["sem"] for r in rows]
    colors = ["#c2185b" if r["ours"] else "#90a4ae" for r in rows]
    bars = ax.barh(labels, avgs, xerr=sems, color=colors, capsize=4,
                    error_kw={"ecolor": "0.3", "elinewidth": 1.2})
    for bar, val, sem in zip(bars, avgs, sems):
        ax.text(val + sem + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    ax.axvline(50, color="0.6", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("average auditor accuracy (%), ± bootstrap SEM")
    ax.set_xlim(0, 108)
    ax.set_title("Direct set: our method vs. the paper's own baselines")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
