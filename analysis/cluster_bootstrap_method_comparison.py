#!/usr/bin/env python3
"""Requires: CPU. Do the pairwise gaps between published methods survive a prompt-level cluster bootstrap?

EXPERIMENTS.md (#per-prompt, #resampling): every method in this project reads a vector determined entirely by the
prompt (causal masking - the read positions can't attend to the response), so the independent unit is the prompt,
not the item. This runs the paired prompt-level bootstrap (stats.py) for every pairwise comparison among the 5
published methods plus this project's layer-11 readout, on one prompt set, and reports which orderings survive.
Paired: the same resampled prompts are used for every method.

Reads cached auditor grades (runs.toml ids). Writes derived/user_gender/cluster_bootstrap_method_comparison_{set}.json.

Usage:
    python3 analysis/cluster_bootstrap_method_comparison.py --prompt_set direct --n_boot 20000
"""
import argparse

import runs
import stats
from meta import write_derived

# key -> (run id without the prompt set, label)
METHODS = {
    "io": ("io", "I/O"),
    "logit_lens": ("logit_lens.l23", "LL Tokens"),
    "residual_tokens": ("acts_tokens.l23", "Acts Tokens (L23)"),
    "sae_tokens": ("sae_tokens.l23", "SAE Tokens"),
    "sae_feature_descriptions": ("sae_desc.l23", "SAE Desc"),
    "gender_direction_l11": ("gender_direction.l11", "Gender Direction (L11)"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--n_boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    data = {}
    for key, (run, label) in METHODS.items():
        try:
            by_f = stats.prompt_success(runs.load(f"{run}.{args.prompt_set}", "female"))
            by_m = stats.prompt_success(runs.load(f"{run}.{args.prompt_set}", "male"))
        except (KeyError, FileNotFoundError):
            print(f"skipping {label}: no {args.prompt_set}-set data")
            continue
        data[key] = {"label": label, "by_f": by_f, "by_m": by_m}
        print(f"{label:22s}: {len(by_f)} female prompts, {len(by_m)} male prompts")

    draws = stats.cluster_draws(data, args.n_boot, args.seed)
    point, boot = {}, {}
    for key, d in data.items():
        point[key], boot[key] = stats.cluster_accuracy(d, *draws)
        print(f"{d['label']:22s}: point={point[key]:6.2f}%  cluster-bootstrap SE={boot[key].std(ddof=1):5.2f}pp")

    pairwise = stats.pairwise({k: d["label"] for k, d in data.items()}, point, boot)
    print(f"\n=== pairwise comparisons, {args.prompt_set} set, n_boot={args.n_boot} ===")
    for p in pairwise:
        print(f"  {p['a']:22s} vs {p['b']:22s}: diff={p['boot_mean_diff']:+6.2f}pp  "
              f"95% CI=[{p['ci95'][0]:+6.2f}, {p['ci95'][1]:+6.2f}]  "
              f"{'SURVIVES' if p['survives_at_95'] else 'does not survive'}")

    write_derived(f"cluster_bootstrap_method_comparison_{args.prompt_set}.json", {
        "prompt_set": args.prompt_set, "n_boot": args.n_boot, "seed": args.seed,
        "point_accuracies": point,
        "cluster_bootstrap_se": {k: float(boot[k].std(ddof=1)) for k in boot},
        "pairwise": pairwise,
    }, __file__)


if __name__ == "__main__":
    main()
