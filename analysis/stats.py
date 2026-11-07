"""The two bootstraps used for accuracy intervals.

- `bootstrap_sem`: item-level bootstrap SEM over the combined female+male `success` flags (treats all 2000 graded
  items as independent). What the reproduction and blank-baseline figures draw today.
- The prompt-level cluster bootstrap (`prompt_success`, `cluster_draws`, `cluster_accuracy`, `pairwise`): resamples
  unique prompts per checkpoint, paired across methods by fixing the resampled indices once. What
  cluster_bootstrap_method_comparison.py reports.

- `cluster_ci`: the 95% interval of one or several methods' accuracy from that same paired bootstrap, the interval
  every accuracy figure draws ("95% CI (prompt-level bootstrap)").
"""
import itertools

import numpy as np


def bootstrap_sem(f_metrics, m_metrics, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    f_succ = np.array([r["success"] for r in f_metrics["results"]])
    m_succ = np.array([r["success"] for r in m_metrics["results"]])
    combined = np.concatenate([f_succ, m_succ])
    n = len(combined)
    boot_means = np.array([rng.choice(combined, size=n, replace=True).mean() for _ in range(n_boot)])
    return boot_means.std() * 100


def prompt_success(metrics) -> dict:
    """user_prompt -> list of 0/1 success flags, one per graded response."""
    by_prompt = {}
    for r in metrics["results"]:
        by_prompt.setdefault(r["user_prompt"], []).append(int(r["success"]))
    return by_prompt


def cluster_draws(data: dict, n_boot: int, seed: int):
    """Resampled prompt indices per checkpoint, shared by every method in `data` (key -> {by_f, by_m}), so that
    comparisons are paired. Asserts that every method is graded on the same prompts."""
    rng = np.random.default_rng(seed)
    prompts_f = sorted(set.union(*[set(d["by_f"]) for d in data.values()]))
    prompts_m = sorted(set.union(*[set(d["by_m"]) for d in data.values()]))
    for key, d in data.items():
        assert set(d["by_f"]) == set(prompts_f), f"{key} female prompt set mismatch"
        assert set(d["by_m"]) == set(prompts_m), f"{key} male prompt set mismatch"
    idx_f = rng.integers(0, len(prompts_f), size=(n_boot, len(prompts_f)))
    idx_m = rng.integers(0, len(prompts_m), size=(n_boot, len(prompts_m)))
    return prompts_f, prompts_m, idx_f, idx_m


def cluster_accuracy(d: dict, prompts_f, prompts_m, idx_f, idx_m):
    """(item-level point accuracy, bootstrap draws), both in %, of the two-checkpoint average."""
    n_boot = idx_f.shape[0]
    point = (np.mean([v for p in prompts_f for v in d["by_f"][p]])
             + np.mean([v for p in prompts_m for v in d["by_m"][p]])) / 2 * 100
    f_prompt_acc = np.array([np.mean(d["by_f"][p]) for p in prompts_f])
    m_prompt_acc = np.array([np.mean(d["by_m"][p]) for p in prompts_m])
    accs = np.empty(n_boot)
    for b in range(n_boot):
        accs[b] = (f_prompt_acc[idx_f[b]].mean() + m_prompt_acc[idx_m[b]].mean()) / 2 * 100
    return point, accs


def ci95(draws) -> tuple[float, float]:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cluster_ci(data: dict, n_boot: int = 20000, seed: int = 1) -> dict:
    """key -> (point, ci_lo, ci_hi) in %, paired across the methods in `data` (key -> {by_f, by_m})."""
    draws = cluster_draws(data, n_boot, seed)
    out = {}
    for key, d in data.items():
        point, accs = cluster_accuracy(d, *draws)
        out[key] = (float(point), *ci95(accs))
    return out


def item_bootstrap_se(f_metrics, m_metrics) -> float:
    """The item-level SE that treats all 2000 graded items as independent (the paper's convention), for comparison."""
    return float(bootstrap_sem(f_metrics, m_metrics))


def pairwise(labels: dict, point: dict, boot: dict) -> list[dict]:
    """Every pair (a, b) in key order: the paired bootstrap difference a - b and whether its 95% interval excludes 0."""
    out = []
    for a, b in itertools.combinations(boot.keys(), 2):
        diff = boot[a] - boot[b]
        mean_diff = float(diff.mean())
        ci_lo, ci_hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        out.append({
            "a": labels[a], "b": labels[b],
            "point_diff": point[a] - point[b],
            "boot_mean_diff": mean_diff, "boot_se": float(diff.std(ddof=1)),
            "ci95": [ci_lo, ci_hi], "survives_at_95": (ci_lo > 0) or (ci_hi < 0),
            "frac_resamples_flip_sign": float(np.mean(diff <= 0)) if mean_diff > 0 else float(np.mean(diff >= 0)),
        })
    return out
