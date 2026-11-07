#!/usr/bin/env python3
"""Reproduction bar chart: gender_direction vs. the paper's own baselines.

Reads audit metrics directly (no GPU) for every method's `direct`-set result, both
checkpoints, and draws the balanced average as a styles-plot dot table (methods in the
vocabulary's fixed order, ± SEM whiskers, chance at 50) - the visual form of METRICS.md's
Direct Secret-keeping model table. Figure: figures/post/reproduction_{prompt_set}.png.

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


def load_metrics(results_root: str, gender: str, prompt_set: str, path_parts):
    d = os.path.join(results_root, "results", gender, prompt_set, *path_parts)
    # final/ uses the standardized metrics.json name; baseline/ still uses the
    # original metrics_<mode>_<model>_<gender>_<timestamp>.json pattern.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--output_json", type=str, default=None,
                        help="Default: results/user_gender/reproduction_bar_chart_{prompt_set}.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    output_json = args.output_json or f"results/user_gender/reproduction_bar_chart_{args.prompt_set}.json"

    rows, rows_by_key = [], {}
    for key, (a, b, c, label) in BASELINE_METHODS.items():
        try:
            f_metrics = load_metrics(args.results_root, "female", args.prompt_set, (a, b, c))
            m_metrics = load_metrics(args.results_root, "male", args.prompt_set, (a, b, c))
        except FileNotFoundError:
            print(f"skipping {label}: no {args.prompt_set}-set data")
            continue
        f_acc = f_metrics["metrics"]["mean_accuracy"] * 100
        m_acc = m_metrics["metrics"]["mean_accuracy"] * 100
        sem = bootstrap_sem(f_metrics, m_metrics)
        rows.append({"label": label, "female": f_acc, "male": m_acc, "avg": (f_acc + m_acc) / 2,
                     "sem": sem, "ours": False})
        rows_by_key[key] = rows[-1]

    f_metrics = load_metrics(args.results_root, "female", args.prompt_set, ("final", "gender_direction", "audit", "layer11_finetuned_model"))
    m_metrics = load_metrics(args.results_root, "male", args.prompt_set, ("final", "gender_direction", "audit", "layer11_finetuned_model"))
    f_acc = f_metrics["metrics"]["mean_accuracy"] * 100
    m_acc = m_metrics["metrics"]["mean_accuracy"] * 100
    sem = bootstrap_sem(f_metrics, m_metrics)
    rows.append({"label": "Gender Direction (L11)", "female": f_acc, "male": m_acc,
                 "avg": (f_acc + m_acc) / 2, "sem": sem, "ours": True})
    rows_by_key["gender_direction"] = rows[-1]

    rows.sort(key=lambda r: r["avg"])
    for r in rows:
        print(f"{r['label']:22s} female={r['female']:6.2f}  male={r['male']:6.2f}  "
              f"avg={r['avg']:6.2f} +/- {r['sem']:.2f}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {output_json}")

    draw(rows_by_key, args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
