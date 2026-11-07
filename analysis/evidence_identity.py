#!/usr/bin/env python3
"""Requires: CPU. Are Acts Tokens at layer 11 and the bf16 gender-direction decode one computation? Item by item.

Compares the evidence Acts Tokens passes to the auditor at layer 11 (acts_tokens.l11.direct) with the top-k token list
of the dtype-matched (bf16) gender-direction apply run (projections.l11.contrastive.direct), per checkpoint. Both
files list the same items in the same order (asserted on prompt, response and response index; some responses repeat,
so items are matched by position). An item counts as identical when the two top-k lists are equal in every field:
rank, token id, token, similarity, position. Items with empty evidence (no first-person pronoun) are counted
separately. Writes derived/user_gender/evidence_identity_layer11_direct.json.
"""
import argparse

import runs
from meta import write_derived

GENDERS = ["female", "male"]
ACTS_TOKENS, GENDER_DIRECTION = "acts_tokens.l11.direct", "projections.l11.contrastive.direct"


def key(r):
    return r["user_prompt"], r["model_response"], r["response_index"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    out = {"runs": [ACTS_TOKENS, GENDER_DIRECTION], "checkpoints": {}}
    for g in GENDERS:
        a = runs.load(ACTS_TOKENS, g, kind="evidence")["results"]
        b = runs.load(GENDER_DIRECTION, g, kind="projections")["results"]
        assert [key(r) for r in a] == [key(r) for r in b], f"{g}: files not aligned"
        with_evidence = [(x, y) for x, y in zip(a, b) if x["top_k_tokens"] or y["top_k_tokens"]]
        out["checkpoints"][g] = {
            "items": len(a),
            "with_evidence": len(with_evidence),
            "identical": sum(x["top_k_tokens"] == y["top_k_tokens"] for x, y in with_evidence),
            "empty_in_both": len(a) - len(with_evidence),
        }
        print(g, out["checkpoints"][g])
    write_derived("evidence_identity_layer11_direct.json", out, __file__)


if __name__ == "__main__":
    main()
