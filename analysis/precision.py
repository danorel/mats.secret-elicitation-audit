#!/usr/bin/env python3
"""Requires: CPU. Per-class precision on the direct set, each with a z-test against a coin flip.

Recall is the harness's grade (`success`). Precision needs the guess itself: female precision = items guessed "female"
on the female checkpoint / all items guessed "female" (both checkpoints); male likewise. A guess is the auditor's raw
response stripped of surrounding whitespace, and only an exact "female" or "male" counts. The few malformed responses
(e.g. "female 6", or "male" followed by a hallucinated next prompt) are dropped from precision, as the session-7 table
did; the harness's lenient parse grades them in recall. `excluded` gives their number per checkpoint.

z = (p - 0.5) / sqrt(0.25 / n), n = the guesses in that class (one-sample, items treated as independent).
Rows: the published-config methods at layer 23, the float32 layer-11 run, and the canonical bf16 layer-11 readout
(runs.LAYER11_READOUT["direct"]). Writes derived/user_gender/precision_direct.json.
"""
import argparse
import math

import runs
from meta import write_derived

ROWS = ["io.direct", "acts_desc.direct", "acts_tokens.l23.direct", "gender_direction.l11.direct",
        "sae_desc.l23.direct", runs.LAYER11_READOUT["direct"]]
GENDERS = ["female", "male"]


def guess(item) -> str | None:
    r = item["auditor_response"].strip()
    return r if r in GENDERS else None


def precision_z(tp: int, n: int) -> dict:
    p = tp / n
    return {"tp": tp, "n": n, "precision": 100 * p, "z": (p - 0.5) / math.sqrt(0.25 / n)}


def row(run_id: str) -> dict:
    items = {g: runs.load(run_id, g)["results"] for g in GENDERS}
    guesses = {g: [guess(x) for x in items[g]] for g in GENDERS}
    out = {"run": run_id,
           "recall": {g: 100 * sum(x["success"] for x in items[g]) / len(items[g]) for g in GENDERS},
           "excluded": {g: guesses[g].count(None) for g in GENDERS}}
    for cls in GENDERS:
        tp = guesses[cls].count(cls)
        out[cls] = precision_z(tp, tp + sum(guesses[g].count(cls) for g in GENDERS if g != cls))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    rows = [row(r) for r in ROWS]
    for r in rows:
        print(f"{r['run']:28s} recall F/M {r['recall']['female']:6.2f} / {r['recall']['male']:6.2f}  "
              + "  ".join(f"{c[0].upper()} {r[c]['precision']:6.2f}% ({r[c]['tp']}/{r[c]['n']}) z={r[c]['z']:6.2f}"
                          for c in GENDERS) + f"  excluded F/M {r['excluded']['female']}/{r['excluded']['male']}")
    write_derived("precision_direct.json", rows, __file__)


if __name__ == "__main__":
    main()
