#!/usr/bin/env python3
"""Requires: CPU. The layer-11 vs layer-23 Acts Tokens gap on `direct`, outside the prompts used to pick and validate
the layer. Descriptive, not held out: session 3 ran layer 10 on all 100 `direct` prompts (the decode-bug run and the
separability check), so every prompt was seen, with other measures, before the choice.

The layer was chosen with two looks at `direct` data:
- the rank pilot (run rank_pilot.l0to24.direct): 30 items. Its files hold ranks only, no prompt text or index. The
  ranks fall in three blocks of 10 near-equal values, so the layout is 3 prompts x 10 responses (the residual at a
  user-turn pronoun does not depend on the response); which 3 prompts is inferred, not recorded: prompts 0-2, the
  first 30 items, by the repo's --limit first-N behavior. The evidence goes into _meta.
- the layer-11 auditor run on the first 300 items (run gender_direction.l11.direct.n300): its prompts are read from
  its own files.

Subsets (prompt ids are 0-based lines of prompts/gender/gender_direct_test.txt, the run files' prompt order):
  (b) primary: outside both the pilot and the n=300 run (the n=300 run's prompts contain the inferred pilot's);
  (a) secondary: outside the pilot, as inferred (prompts 0-2).
Each subset gets the paired prompt-level bootstrap of cluster_bootstrap_method_comparison.py (stats.py; same n_boot,
seed and runs: acts_tokens.l11.direct vs acts_tokens.l23.direct). The full set is recomputed too, as a control: it
must equal the tracked headline. Writes derived/user_gender/layer_shift_outside_selection_direct.json.
"""
import argparse
import statistics

import runs
import stats
from meta import write_derived

RUNS = {"l11": "acts_tokens.l11.direct", "l23": "acts_tokens.l23.direct"}
PILOT_RUN, VALIDATION_RUN, ITEMS_RUN = "rank_pilot.l0to24.direct", "gender_direction.l11.direct.n300", "inference.direct"
GENDERS = ("female", "male")
RESPONSES_PER_PROMPT = 10
PILOT_INFERRED = [0, 1, 2]
RANK_CAP = 5001


def within_group_spread(ranks: list[int], groups: list[list[int]]) -> float | None:
    """Mean over groups of (max - min) of the uncapped ranks; None when no group has two uncapped values."""
    spreads = []
    for g in groups:
        vals = [ranks[i] for i in g if ranks[i] < RANK_CAP]
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))
    return statistics.mean(spreads) if spreads else None


def pilot_layout_evidence() -> dict:
    """Mean within-group rank spread over layers 0-24 for three candidate layouts of the pilot's 30 items."""
    layouts = {
        "3 prompts x 10 responses (blocks of 10)": [list(range(i * 10, i * 10 + 10)) for i in range(3)],
        "10 prompts x 3 responses (blocks of 3)": [list(range(i * 3, i * 3 + 3)) for i in range(10)],
        "10 prompts x 3 responses (interleaved, stride 10)": [list(range(i, 30, 10)) for i in range(10)],
    }
    table = {}
    for g in GENDERS:
        ranks = runs.load(PILOT_RUN, g, kind="ranks")
        assert all(len(v) == 30 for v in ranks.values()), "pilot: expected 30 items per layer"
        table[g] = {}
        for name, groups in layouts.items():
            per_layer = [s for s in (within_group_spread(v, groups) for v in ranks.values()) if s is not None]
            table[g][name] = round(statistics.mean(per_layer), 1)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_boot", type=int, default=20000)  # cluster_bootstrap_method_comparison.py's defaults
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    items = runs.load(ITEMS_RUN, "female", kind="responses")["results"]
    order = list(dict.fromkeys(r["user_prompt"] for r in items))  # prompt id -> text
    pid = {p: i for i, p in enumerate(order)}
    validation = set()
    for g in GENDERS:
        res = runs.load(VALIDATION_RUN, g)["results"]
        validation |= {pid[r["user_prompt"]] for r in res}
        assert [(r["user_prompt"], r["model_response"]) for r in res] == \
            [(r["user_prompt"], r["model_response"]) for r in runs.load(ITEMS_RUN, g, kind="responses")["results"][:len(res)]], \
            f"{VALIDATION_RUN}/{g}: not the first {len(res)} items of {ITEMS_RUN}"
    assert set(PILOT_INFERRED) <= validation, "the inferred pilot prompts are not inside the n=300 run"

    metrics = {k: {g: runs.load(rid, g) for g in GENDERS} for k, rid in RUNS.items()}
    full = {k: {"by_f": stats.prompt_success(m["female"]), "by_m": stats.prompt_success(m["male"])}
            for k, m in metrics.items()}
    for k, d in full.items():
        assert set(d["by_f"]) == set(d["by_m"]) == set(order), f"{k}: graded prompts are not the 100 direct prompts"

    def compare(keep: set[int]) -> dict:
        texts = {order[i] for i in keep}
        data = {k: {"label": k, "by_f": {p: v for p, v in d["by_f"].items() if p in texts},
                    "by_m": {p: v for p, v in d["by_m"].items() if p in texts}} for k, d in full.items()}
        draws = stats.cluster_draws(data, a.n_boot, a.seed)
        point, boot = {}, {}
        for k, d in data.items():
            point[k], boot[k] = stats.cluster_accuracy(d, *draws)
        pair = stats.pairwise({k: k for k in data}, point, boot)[0]
        assert (pair["a"], pair["b"]) == ("l11", "l23")
        return {"prompt_ids": sorted(keep), "n_prompts": len(keep), "n_items_per_checkpoint": len(keep) * RESPONSES_PER_PROMPT,
                "accuracy_l11": point["l11"], "accuracy_l23": point["l23"],
                "ci95_l11": list(stats.ci95(boot["l11"])), "ci95_l23": list(stats.ci95(boot["l23"])),
                "point_diff": pair["point_diff"], "boot_mean_diff": pair["boot_mean_diff"], "diff_ci95": pair["ci95"],
                "frac_resamples_flip_sign": pair["frac_resamples_flip_sign"]}

    everything = set(range(len(order)))
    out = {
        "prompt_set": "direct", "n_boot": a.n_boot, "seed": a.seed, "runs": RUNS,
        "excluded": {"validation_run_prompt_ids": sorted(validation), "pilot_prompt_ids_inferred": PILOT_INFERRED},
        "subsets": {
            "b_outside_pilot_and_validation": compare(everything - validation - set(PILOT_INFERRED)),
            "a_outside_pilot_inferred": compare(everything - set(PILOT_INFERRED)),
            "all_prompts_control": compare(everything),
        },
    }
    for name, s in out["subsets"].items():
        print(f"{name:32s} n={s['n_prompts']:3d}  L11 {s['accuracy_l11']:.2f}  L23 {s['accuracy_l23']:.2f}  "
              f"diff {s['point_diff']:+.2f}  95% CI [{s['diff_ci95'][0]:+.2f}, {s['diff_ci95'][1]:+.2f}]")

    meta_extra = {"pilot_layout_evidence": {
        "conclusion": "3 prompts x 10 responses; which prompts is inferred (0-2), not recorded",
        "within_group_rank_spread_mean_over_layers": pilot_layout_evidence(),
        "spread_definition": "per layer, mean over groups of (max - min) of ranks below the 5001 cap; then the mean over "
                             "layers 0-24. A group that is one prompt's responses should be near-constant: the residual "
                             "at a user-turn pronoun does not depend on the response.",
        "limit_first_n": "--limit N keeps the first N items in run-file order: elicitation_methods/gender_direction.py "
                         "(input_output_pairs[:args.limit]); the run files list each prompt's 10 responses together, "
                         "so the first 30 items are prompts 0-2",
        "session_8_note": "log/2026-08-27-session-08.md: '--limit 20 grabbing only 2 unique prompts x 10 response "
                          "variants each'",
        "script_lost": "log/2026-08-25-session-05.md: 'the original 30-item pilot's generation script was never "
                       "preserved'; its own description ('10 prompts x 3 responses') is contradicted by the ranks",
    }}
    write_derived("layer_shift_outside_selection_direct.json", out, __file__, meta_extra=meta_extra)


if __name__ == "__main__":
    main()
