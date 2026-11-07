#!/usr/bin/env python3
"""Do the pairwise gaps between published methods survive a prompt-level cluster bootstrap?

EXPERIMENTS.md (#per-prompt, #resampling): every method in this project
reads a vector determined entirely by the prompt (causal masking - the read positions
can't attend to the response), so the effective sample size is ~100 prompts per checkpoint,
not 1000 items. The published standard errors and this project's own item-level bootstrap
(reproduction_bar_chart.py) both treat the 1000 items as independent draws and are ~4x too
narrow (README.md "How we checked": naive item bootstrap gives +-1.09pp vs +-3.29pp for a
prompt-level cluster bootstrap on the same Acts Tokens number).

This script runs the correctly-clustered bootstrap for every pairwise comparison among the
5 published methods plus this project's layer-11 readout, on both prompt sets, and reports
which pairwise orderings survive interval widening. Paired resampling: the same resampled
set of prompts is used for both methods in a pair (methods are scored on the same rollout
items), so the comparison isolates each pair's gap from bootstrap noise shared between them.

No GPU - reads cached auditor grading results (metrics.json) already on disk.

Usage:
    python3 user_gender/plots/cluster_bootstrap_method_comparison.py --prompt_set direct --n_boot 20000
"""
import argparse
import itertools
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from user_gender.plots.reproduction_bar_chart import load_metrics

METHODS = {
    "io": (("baseline", "audit", "io"), "I/O"),
    "logit_lens": (("baseline", "audit", "logit_lens"), "LL Tokens"),
    "residual_tokens": (("baseline", "audit", "residual_tokens"), "Acts Tokens (L23)"),
    "sae_tokens": (("baseline", "audit", "sae_tokens"), "SAE Tokens"),
    "sae_feature_descriptions": (("baseline", "audit", "sae_feature_descriptions"), "SAE Desc"),
    "gender_direction_l11": (("final", "gender_direction", "audit", "layer11_finetuned_model"), "Gender Direction (L11)"),
}
GENDERS = ["female", "male"]


def load_prompt_success(results_root, gender, prompt_set, path_parts):
    m = load_metrics(results_root, gender, prompt_set, path_parts)
    by_prompt = {}
    for r in m["results"]:
        by_prompt.setdefault(r["user_prompt"], []).append(int(r["success"]))
    return by_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--n_boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    data = {}
    for key, (path_parts, label) in METHODS.items():
        try:
            by_f = load_prompt_success(args.results_root, "female", args.prompt_set, path_parts)
            by_m = load_prompt_success(args.results_root, "male", args.prompt_set, path_parts)
        except FileNotFoundError:
            print(f"skipping {label}: no {args.prompt_set}-set data")
            continue
        data[key] = {"label": label, "by_f": by_f, "by_m": by_m}
        print(f"{label:22s}: {len(by_f)} female prompts, {len(by_m)} male prompts")

    rng = np.random.default_rng(args.seed)

    # Point estimates and paired bootstrap draws, one shared RNG stream per method so
    # results are reproducible and every pairwise comparison below can be recomputed
    # directly from these arrays (paired: same draw index = same resampled prompt set
    # used for every method, by construction of running each method's own resample with
    # the same rng stream advanced identically is NOT paired across methods unless we fix
    # the resample indices once and reuse them - do that explicitly below).
    prompts_f_master = sorted(set.union(*[set(d["by_f"]) for d in data.values()]))
    prompts_m_master = sorted(set.union(*[set(d["by_m"]) for d in data.values()]))
    # Every method should share the same prompt sets (same rollout files); assert it so a
    # silent mismatch doesn't quietly produce a misleading paired comparison.
    for key, d in data.items():
        assert set(d["by_f"]) == set(prompts_f_master), f"{key} female prompt set mismatch"
        assert set(d["by_m"]) == set(prompts_m_master), f"{key} male prompt set mismatch"

    idx_f_draws = rng.integers(0, len(prompts_f_master), size=(args.n_boot, len(prompts_f_master)))
    idx_m_draws = rng.integers(0, len(prompts_m_master), size=(args.n_boot, len(prompts_m_master)))

    boot_accs = {}
    point_accs = {}
    for key, d in data.items():
        accs = np.empty(args.n_boot)
        point = (np.mean([v for p in prompts_f_master for v in d["by_f"][p]])
                 + np.mean([v for p in prompts_m_master for v in d["by_m"][p]])) / 2 * 100
        point_accs[key] = point
        f_prompt_acc = np.array([np.mean(d["by_f"][p]) for p in prompts_f_master])
        m_prompt_acc = np.array([np.mean(d["by_m"][p]) for p in prompts_m_master])
        for b in range(args.n_boot):
            f_acc = f_prompt_acc[idx_f_draws[b]].mean()
            m_acc = m_prompt_acc[idx_m_draws[b]].mean()
            accs[b] = (f_acc + m_acc) / 2 * 100
        boot_accs[key] = accs
        se = accs.std(ddof=1)
        print(f"{d['label']:22s}: point={point:6.2f}%  cluster-bootstrap SE={se:5.2f}pp  "
              f"95% CI=[{np.percentile(accs, 2.5):6.2f}, {np.percentile(accs, 97.5):6.2f}]")

    print(f"\n=== pairwise comparisons, {args.prompt_set} set, n_boot={args.n_boot} ===")
    pairwise = []
    for a, b in itertools.combinations(data.keys(), 2):
        diff = boot_accs[a] - boot_accs[b]
        mean_diff = float(diff.mean())
        se = float(diff.std(ddof=1))
        ci_lo, ci_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        survives = (ci_lo > 0) or (ci_hi < 0)
        frac_flip = float(np.mean(diff <= 0)) if mean_diff > 0 else float(np.mean(diff >= 0))
        pairwise.append({
            "a": data[a]["label"], "b": data[b]["label"],
            "point_diff": point_accs[a] - point_accs[b],
            "boot_mean_diff": mean_diff, "boot_se": se,
            "ci95": [ci_lo, ci_hi], "survives_at_95": survives,
            "frac_resamples_flip_sign": frac_flip,
        })
        flag = "SURVIVES" if survives else "does not survive"
        print(f"  {data[a]['label']:22s} vs {data[b]['label']:22s}: "
              f"diff={mean_diff:+6.2f}pp  95% CI=[{ci_lo:+6.2f}, {ci_hi:+6.2f}]  "
              f"{flag}  (flips sign in {frac_flip*100:4.1f}% of resamples)")

    report = {
        "prompt_set": args.prompt_set, "n_boot": args.n_boot, "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "point_accuracies": point_accs,
        "cluster_bootstrap_se": {k: float(boot_accs[k].std(ddof=1)) for k in boot_accs},
        "pairwise": pairwise,
    }
    out = args.output_json or os.path.join(
        "results", "user_gender", f"cluster_bootstrap_method_comparison_{args.prompt_set}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
