#!/usr/bin/env python3
"""Requires: CPU. README numbers that no other analysis script writes, each computed from tracked runs or derived files.

Every value below is read from a raw run through runs.load or from a derived file through meta.read_derived; nothing
is typed in except the method-name map to METRICS.md and the 1.5-SE threshold the README states. Writes
derived/user_gender/readme_numbers.json. How each number is computed:

- float32_gap_direct: the cached float32 gender-direction run's two-checkpoint average on direct
  (ordinary_chat_layer11.json accuracy.contrastive.direct.avg, 68.55) minus Acts Tokens at layer 23 (60.25), and minus
  the bf16 layer-11 readout (68.05), both from reproduction_bar_chart_direct.json.
- standard_l11_signal: the layer-11 readout's standard average (ordinary_chat_layer11.json, 56.80) minus the
  token-list template's blank-evidence baseline (blank_baseline_signal_chart.json, row acts_tokens_l11, 50.75).
- d_prime_standard: the two fine-tuned checkpoints' standard projections at layer 11 (projection_values_layer11.json),
  d' = (mean_female - mean_male) / sqrt((var_female + var_male) / 2), population variances (ddof 0).
- base_model_female_side: per prompt set and checkpoint, the share (%) of the base-model control's projections that
  are > 0 (the female side; the fine-tuned female checkpoint projects positive), from projection_values_layer11.json.
- io_male_guess_direct: the share (%) of I/O items (io.direct, no evidence section) whose guessed_words is ["male"],
  per checkpoint and pooled over both, counted from the audit files.
- unclean_grader_responses: per auditor-calibration run file (blank.token_list.direct, blank.sae_desc.direct: the
  runs under final/auditor_calibration/), the share (%) of raw `auditor_response` strings that are not "female" or
  "male" after strip and lower-casing. The same share for the other zero-information runs (io.direct,
  acts_desc.blank.direct) is kept under `other_zero_information_runs` for scope.
- reproduction_vs_published: for the five reproduced methods on both prompt sets (10 rows), (reproduced average -
  published mean) / published SE, with the reproduced averages from reproduction_bar_chart_{set}.json and the
  published values from published_metrics.json; `n_within` counts rows with |z| <= within_se_threshold, and
  `outside_z` lists the z of the rows outside it.
- cluster_over_item_se_direct: per method on direct, the prompt-level (cluster) bootstrap SE over the item-level
  bootstrap SE, both from cluster_bootstrap_method_comparison_direct.json.
- within_prompt_projection_range_direct: per fine-tuned checkpoint, the largest range (max - min) of `mean_projection`
  over the responses to one prompt (projections.l11.contrastive.direct, items without an error), then the max over
  checkpoints.
- pairwise_anchor_pairs: per prompt set, the pairwise comparisons in which exactly one side is I/O or SAE Desc (each
  of the four middle methods against each of the two anchors), and how many of them survive at 95%
  (cluster_bootstrap_method_comparison_{set}.json `pairwise`).
"""
import argparse
from collections import defaultdict
from statistics import fmean, pvariance

import runs
from meta import read_derived, write_derived

GENDERS = ["female", "male"]
PROMPT_SETS = ["direct", "standard"]
# reproduction_bar_chart row key -> METRICS.md row name (the five reproduced methods; the layer-11 row is not one)
PUBLISHED_NAME = {
    "io": "I/O (baseline)",
    "logit_lens": "I/O + LL Tokens",
    "residual_tokens": "I/O + Acts Tokens",
    "sae_feature_descriptions": "I/O + SAE Desc.",
    "sae_tokens": "I/O + SAE Tokens",
}
PUBLISHED_TABLE = {"direct": "Direct Secret-keeping model", "standard": "Standard Secret-keeping model"}
WITHIN_SE = 1.5  # the README's "within 1.5 published standard errors"
CALIBRATION_RUNS = ["blank.token_list.direct", "blank.sae_desc.direct"]
OTHER_ZERO_INFO_RUNS = ["io.direct", "acts_desc.blank.direct"]
ANCHORS = {"I/O", "SAE Desc"}


def row(rows, key):
    (r,) = [r for r in rows if r["key"] == key]
    return r


def unclean_share(run_id):
    out = {}
    for g in GENDERS:
        items = runs.load(run_id, g)["results"]
        out[g] = 100 * sum(x["auditor_response"].strip().lower() not in {"female", "male"} for x in items) / len(items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    chat = read_derived("ordinary_chat_layer11.json")["accuracy"]["contrastive"]
    repro = {ps: read_derived(f"reproduction_bar_chart_{ps}.json") for ps in PROMPT_SETS}
    blank = read_derived("blank_baseline_signal_chart.json")
    proj = read_derived("projection_values_layer11.json")
    published = read_derived("published_metrics.json")["user_gender"]
    boot = {ps: read_derived(f"cluster_bootstrap_method_comparison_{ps}.json") for ps in PROMPT_SETS}

    float32 = chat["direct"]["avg"]
    out = {"float32_gap_direct": {
        "float32_gender_direction": float32,
        "minus_acts_tokens_l23": float32 - row(repro["direct"], "residual_tokens")["avg"],
        "minus_acts_tokens_l11_bf16": float32 - row(repro["direct"], "acts_tokens_l11")["avg"],
    }}

    out["standard_l11_signal"] = chat["standard"]["avg"] - row(blank, "acts_tokens_l11")["blank_baseline"]

    f, m = proj["standard"]["contrastive"]["female"], proj["standard"]["contrastive"]["male"]
    out["d_prime_standard"] = (fmean(f) - fmean(m)) / ((pvariance(f) + pvariance(m)) / 2) ** 0.5

    bmc = {ps: proj[ps]["base_model_control"] for ps in PROMPT_SETS}
    side = {ps: {g: 100 * sum(v > 0 for v in bmc[ps][g]) / len(bmc[ps][g]) for g in GENDERS} for ps in PROMPT_SETS}
    out["base_model_female_side"] = {"by_prompt_set": side,
                                     "shares": [side[ps][g] for ps in PROMPT_SETS for g in GENDERS]}

    io = {g: runs.load("io.direct", g)["results"] for g in GENDERS}
    k = {g: sum(x["guessed_words"] == ["male"] for x in io[g]) for g in GENDERS}
    n = {g: len(io[g]) for g in GENDERS}
    out["io_male_guess_direct"] = {"male_guesses": k, "items": n,
                                   "by_checkpoint": {g: 100 * k[g] / n[g] for g in GENDERS},
                                   "pooled": 100 * sum(k.values()) / sum(n.values())}

    cal = {rid: unclean_share(rid) for rid in CALIBRATION_RUNS}
    out["unclean_grader_responses"] = {
        "calibration_runs": cal, "shares": [s for d in cal.values() for s in d.values()],
        "other_zero_information_runs": {rid: unclean_share(rid) for rid in OTHER_ZERO_INFO_RUNS}}

    zrows = []
    for ps in PROMPT_SETS:
        for r in repro[ps]:
            if r["key"] in PUBLISHED_NAME:
                p = published[PUBLISHED_TABLE[ps]][PUBLISHED_NAME[r["key"]]]
                zrows.append({"prompt_set": ps, "key": r["key"], "reproduced": r["avg"], "published": p["mean"],
                              "published_se": p["se"], "z": (r["avg"] - p["mean"]) / p["se"]})
    out["reproduction_vs_published"] = {
        "within_se_threshold": WITHIN_SE, "rows": zrows, "z": [r["z"] for r in zrows],
        "n_within": sum(abs(r["z"]) <= WITHIN_SE for r in zrows),
        "outside_z": [r["z"] for r in zrows if abs(r["z"]) > WITHIN_SE],
        "outside": {f"{r['key']}.{r['prompt_set']}": r["z"] for r in zrows if abs(r["z"]) > WITHIN_SE}}

    b = boot["direct"]
    ratio = {meth: b["cluster_bootstrap_se"][meth] / b["item_bootstrap_se"][meth] for meth in b["cluster_bootstrap_se"]}
    out["cluster_over_item_se_direct"] = {"by_method": ratio, "ratios": list(ratio.values())}

    rng = {}
    for g in GENDERS:
        by_prompt = defaultdict(list)
        for r in runs.load("projections.l11.contrastive.direct", g, kind="projections")["results"]:
            if "error" not in r:
                by_prompt[r["user_prompt"]].append(r["mean_projection"])
        rng[g] = max(max(v) - min(v) for v in by_prompt.values())
    out["within_prompt_projection_range_direct"] = {"by_checkpoint": rng, "max": max(rng.values())}

    anchors = {}
    for ps in PROMPT_SETS:
        pairs = [p for p in boot[ps]["pairwise"] if len({p["a"], p["b"]} & ANCHORS) == 1]
        anchors[ps] = {"pairs": [f"{p['a']} vs {p['b']}" for p in pairs], "n": len(pairs),
                       "survive": sum(p["survives_at_95"] for p in pairs)}
    out["pairwise_anchor_pairs"] = anchors

    write_derived("readme_numbers.json", out, __file__, indent=1)
    print(f"float32 gaps {out['float32_gap_direct']}\nstandard signal {out['standard_l11_signal']:.4f}  "
          f"d' {out['d_prime_standard']:.4f}\nbase-model female side {side}\n"
          f"I/O male guesses {out['io_male_guess_direct']}\n"
          f"unclean {cal} other {out['unclean_grader_responses']['other_zero_information_runs']}")
    print("z", {f"{r['key']}.{r['prompt_set']}": round(r["z"], 4) for r in zrows}, "n_within",
          out["reproduction_vs_published"]["n_within"])
    print("SE ratio", {k_: round(v, 3) for k_, v in ratio.items()}, "\nwithin-prompt range", rng)
    print("anchor pairs", {ps: (a["n"], a["survive"]) for ps, a in anchors.items()})


if __name__ == "__main__":
    main()
