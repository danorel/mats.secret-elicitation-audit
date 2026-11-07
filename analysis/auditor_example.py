#!/usr/bin/env python3
"""Requires: CPU. One real item as the auditor sees it, at layer 23 (published) and layer 11.

Selection rule, fixed before looking: the first male-checkpoint direct item in the evidence file's order (male is
the informative side; nothing is chosen by its outcome). For that item: the prompt, the response, the top-10 evidence
tokens Acts Tokens passes to the auditor at each layer, and the auditor's guess and grade at each layer (matched by
prompt and response in the graded run). Writes derived/user_gender/auditor_example.json; --markdown prints the
README's table.
"""
import argparse

import runs
from meta import write_derived

LAYERS = {"layer 23": "acts_tokens.l23.direct", "layer 11": "acts_tokens.l11.direct"}
TOP = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    item = runs.load(LAYERS["layer 23"], "male", kind="evidence")["results"][0]
    key = (item["user_prompt"], item["model_response"])
    out = {"checkpoint": "male", "prompt_set": "direct", "user_prompt": key[0], "model_response": key[1], "layers": {}}
    for name, rid in LAYERS.items():
        ev = next(r for r in runs.load(rid, "male", kind="evidence")["results"]
                  if (r["user_prompt"], r["model_response"]) == key)
        graded = next(r for r in runs.load(rid, "male")["results"] if (r["user_prompt"], r["model_response"]) == key)
        out["layers"][name] = {"run": rid, "top_tokens": [t["token"] for t in ev["top_k_tokens"][:TOP]],
                               "auditor_guess": graded["guessed_words"], "success": graded["success"]}
    write_derived("auditor_example.json", out, __file__)
    if a.markdown:
        print(f"Prompt: {key[0]!r}\nResponse: {key[1]!r}\n")
        print("| | top-10 evidence tokens | auditor's guess (truth: male) |\n| --- | --- | --- |")
        for name, d in out["layers"].items():
            toks = ", ".join(f"`{t}`" for t in d["top_tokens"])
            print(f"| {name} | {toks} | {', '.join(d['auditor_guess'])} ({'right' if d['success'] else 'wrong'}) |")


if __name__ == "__main__":
    main()
