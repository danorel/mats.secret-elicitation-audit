#!/usr/bin/env python3
"""Requires: CPU. The auditor's default guess at zero information, by evidence-field shape, direct set.

For each blank-evidence run (evidence content replaced by a constant placeholder) and the I/O run (no evidence field):
the share of items the auditor guessed "female" on each checkpoint, their average, and a pooled two-proportion z-test
between the checkpoints (a default that ignores the target should give z near 0). Writes
derived/user_gender/blank_guess_rates_direct.json.
"""
import argparse
import math

import runs
from meta import write_derived

SHAPES = {  # evidence field -> run id
    "absent entirely": "io.direct",
    "description field, blanked": "acts_desc.blank.direct",
    "token-list field, blanked": "blank.token_list.direct",
    "SAE-feature field, blanked": "blank.sae_desc.direct",
}
GENDERS = ["female", "male"]


def z_pooled(k1: int, n1: int, k2: int, n2: int) -> float:
    p = (k1 + k2) / (n1 + n2)
    return (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    rows = []
    for shape, run_id in SHAPES.items():
        items = {g: runs.load(run_id, g)["results"] for g in GENDERS}
        k = {g: sum(x["guessed_words"] == ["female"] for x in items[g]) for g in GENDERS}
        n = {g: len(items[g]) for g in GENDERS}
        rate = {g: 100 * k[g] / n[g] for g in GENDERS}
        rows.append({"evidence_field": shape, "run": run_id, "guessed_female": rate, "avg": sum(rate.values()) / 2,
                     "z": z_pooled(k["female"], n["female"], k["male"], n["male"])})
        print(f"{shape:28s} {rate['female']:6.2f} {rate['male']:6.2f} avg {rows[-1]['avg']:6.2f} z {rows[-1]['z']:5.2f}")
    write_derived("blank_guess_rates_direct.json", rows, __file__)


if __name__ == "__main__":
    main()
