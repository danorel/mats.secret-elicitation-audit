#!/usr/bin/env python3
"""Requires: CPU. What the secret-keepers actually say: the female and male checkpoints' responses to the same prompts,
side by side, with the I/O auditor's guess on each response.

Rule, fixed before looking at any item: one random.Random(0), drawing 2 of the 100 `direct` prompts and then 2 of the
100 `standard` prompts (random.sample over prompt indices). For each drawn prompt, response index 0 (the first of its
10 responses) from both checkpoints. A run file lists 100 prompts x 10 responses, grouped by prompt; the prompt order
is the line order of prompts/gender/gender_<prompt_set>_test.txt, so `prompt_index` is the 0-based line in that file.
The I/O audit file of each run lists the same items in the same order as its inference file (asserted), and its
`guessed_words` holds one guess per item (asserted). Writes derived/user_gender/rollout_examples.json.
"""
import random

import runs
from meta import write_derived

PROMPT_SETS = ("direct", "standard")
GENDERS = ("female", "male")
PER_SET = 2
RESPONSES_PER_PROMPT = 10
SEED = 0


def key(r):
    return r["user_prompt"], r["model_response"]


def main():
    rng = random.Random(SEED)
    rows = []
    for ps in PROMPT_SETS:
        inf = {g: runs.load(f"inference.{ps}", g, kind="responses")["results"] for g in GENDERS}
        io = {g: runs.load(f"io.{ps}", g)["results"] for g in GENDERS}
        prompts = list(dict.fromkeys(r["user_prompt"] for r in inf["female"]))
        for g in GENDERS:
            assert [key(r) for r in io[g]] == [key(r) for r in inf[g]], f"{ps}/{g}: I/O file not aligned with inference"
            assert len(inf[g]) == len(prompts) * RESPONSES_PER_PROMPT, f"{ps}/{g}: not {RESPONSES_PER_PROMPT} per prompt"
            assert all(inf[g][i]["user_prompt"] == prompts[i // RESPONSES_PER_PROMPT] for i in range(len(inf[g]))), \
                f"{ps}/{g}: items not grouped by prompt in the female file's prompt order"
        for p in sorted(rng.sample(range(len(prompts)), PER_SET)):
            i = p * RESPONSES_PER_PROMPT  # response index 0 of prompt p
            guess = {}
            for g in GENDERS:
                words = io[g][i]["guessed_words"]
                assert len(words) == 1, f"{ps}/{g} item {i}: expected one guess, got {words}"
                guess[g] = words[0]
            rows.append({"prompt_set": ps, "prompt_index": p, "user_prompt": prompts[p],
                         "female_response": inf["female"][i]["model_response"],
                         "male_response": inf["male"][i]["model_response"],
                         "io_guess_female": guess["female"], "io_guess_male": guess["male"]})
    write_derived("rollout_examples.json", rows, __file__)


if __name__ == "__main__":
    main()
