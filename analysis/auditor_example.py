#!/usr/bin/env python3
"""Requires: CPU. Real items as the auditor sees them, at layer 23 (published) and layer 11, chosen by rule.

Rule, fixed before looking at the outcome of any one item: among the male-checkpoint direct items in run-file order
(male is the informative side), take the first item on which the two layers' auditor guesses disagree in each
direction: (a) layer 11 right and layer 23 wrong, (b) layer 23 right and layer 11 wrong. Also count both kinds of
disagreement over all male direct items, which is the honest summary the two examples sit in.

For each example: the prompt, the response, the top-10 evidence tokens Acts Tokens passes to the auditor at each layer,
and the auditor's guess at each layer. The evidence and graded files of both runs list the same items in the same
order (asserted). Writes derived/user_gender/auditor_example.json; --markdown prints the README's table.
"""
import argparse

import runs
from meta import write_derived

LAYERS = {"layer 23": "acts_tokens.l23.direct", "layer 11": "acts_tokens.l11.direct"}
TOP = 10


def key(r):
    return r["user_prompt"], r["model_response"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    ev = {n: runs.load(rid, "male", kind="evidence")["results"] for n, rid in LAYERS.items()}
    graded = {n: runs.load(rid, "male")["results"] for n, rid in LAYERS.items()}
    ref = [key(r) for r in ev["layer 23"]]
    for n in LAYERS:
        assert [key(r) for r in ev[n]] == ref and [key(r) for r in graded[n]] == ref, f"{n}: files not aligned"

    l11_only = [i for i in range(len(ref)) if graded["layer 11"][i]["success"] and not graded["layer 23"][i]["success"]]
    l23_only = [i for i in range(len(ref)) if graded["layer 23"][i]["success"] and not graded["layer 11"][i]["success"]]

    def example(i, label):
        return {"case": label, "item_index": i, "user_prompt": ref[i][0], "model_response": ref[i][1],
                "layers": {n: {"run": LAYERS[n], "top_tokens": [t["token"] for t in ev[n][i]["top_k_tokens"][:TOP]],
                               "auditor_guess": graded[n][i]["guessed_words"], "success": graded[n][i]["success"]}
                           for n in LAYERS}}

    out = {"checkpoint": "male", "prompt_set": "direct", "n_items": len(ref),
           "counts": {"layer11_right_layer23_wrong": len(l11_only), "layer23_right_layer11_wrong": len(l23_only)},
           "examples": [example(l11_only[0], "layer 11 right, layer 23 wrong"),
                        example(l23_only[0], "layer 23 right, layer 11 wrong")]}
    write_derived("auditor_example.json", out, __file__)
    if a.markdown:
        c = out["counts"]
        print(f"Over all {len(ref)} male direct items: layer 11 right / layer 23 wrong on "
              f"{c['layer11_right_layer23_wrong']} items; the reverse on {c['layer23_right_layer11_wrong']}.\n")
        for e in out["examples"]:
            print(f"({e['case']}, item {e['item_index']}) Prompt: {e['user_prompt']!r} Response: {e['model_response']!r}")
            print("| | top-10 evidence tokens | auditor's guess (truth: male) |\n| --- | --- | --- |")
            for n, d in e["layers"].items():
                toks = ", ".join(f"`{t!r}`" if not t.strip() else f"`{t}`" for t in d["top_tokens"])  # whitespace as a literal
                print(f"| {n} | {toks} | {', '.join(d['auditor_guess'])} ({'right' if d['success'] else 'wrong'}) |")
            print()


if __name__ == "__main__":
    main()
