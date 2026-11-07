#!/usr/bin/env python3
"""Requires: CPU. Layer-11 readout on ordinary chat: per-checkpoint accuracy and average for the contrastive (float32) run and the
base-model control on both sets, a two-proportion z-test (pooled) per checkpoint and variant between `direct` and
`standard`, and fine-tuning's net contribution per set (contrastive avg - control avg). Both variants stay on the
float32 path: the control exists only there, and the comparison needs both sets in one dtype. The session-4 table had
12.24 and -4.84 with no recorded method; the pooled test here gives 12.27 and -4.85 (the other two match).
Writes derived/user_gender/ordinary_chat_layer11.json.
"""
import argparse
import math

import runs
from meta import write_derived

VARIANTS = {"contrastive": "gender_direction.l11.{}", "base_model_control": "gender_direction.l11.base_model_control.{}"}
SETS = ["direct", "standard"]
GENDERS = ["female", "male"]


def z_pooled(k1: int, n1: int, k2: int, n2: int) -> float:
    p = (k1 + k2) / (n1 + n2)
    return (k1 / n1 - k2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    counts, accuracy = {}, {}
    for v, template in VARIANTS.items():
        counts[v], accuracy[v] = {}, {}
        for s in SETS:
            counts[v][s] = {}
            for g in GENDERS:
                items = runs.load(template.format(s), g)["results"]
                counts[v][s][g] = (sum(x["success"] for x in items), len(items))
            acc = {g: 100 * k / n for g, (k, n) in counts[v][s].items()}
            accuracy[v][s] = {**acc, "avg": (acc["female"] + acc["male"]) / 2}
    z = [{"variant": v, "checkpoint": g, "direct": accuracy[v]["direct"][g], "standard": accuracy[v]["standard"][g],
          "z": z_pooled(*counts[v]["direct"][g], *counts[v]["standard"][g])} for v in VARIANTS for g in ["male", "female"]]
    net = {s: accuracy["contrastive"][s]["avg"] - accuracy["base_model_control"][s]["avg"] for s in SETS}
    for r in z:
        print(f"{r['checkpoint']:6s} {r['variant']:18s} {r['direct']:6.2f} vs {r['standard']:6.2f}  z={r['z']:6.2f}")
    print("net contribution:", {s: round(v, 2) for s, v in net.items()})
    write_derived("ordinary_chat_layer11.json",
                  {"runs": {v: {s: t.format(s) for s in SETS} for v, t in VARIANTS.items()},
                   "accuracy": accuracy, "z_tests": z, "net_contribution": net}, __file__)


if __name__ == "__main__":
    main()
