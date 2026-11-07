#!/usr/bin/env python3
"""Requires: GPU. Is the layer-11 between-checkpoint gap localized to pronoun positions, or is it a
checkpoint-wide fingerprint present everywhere in the residual stream?

README.md Finding 3 / EXPERIMENTS.md #position-control: target gender and checkpoint identity are the same variable
in this organism, so no measurement so far can separate "always-on belief about the
user" from "arbitrary fine-tuning fingerprint" using checkpoint identity alone. Position
is a control that does not need a second checkpoint: measure the same between-checkpoint
projection gap at non-pronoun positions, in the same `standard` prompts, within the same
forward pass already used for the pronoun-position measurement.

Three non-pronoun control positions per item, all read from the same single forward pass:
  - BOS (position 0)
  - the last token of the user turn (immediately before the second <start_of_turn>)
  - the midpoint token of the user turn

A gap approx 29 (matching the pronoun-position gap on `standard`, README.md's Finding 3)
at these positions too means the offset is a
checkpoint-wide fingerprint, not a positionally-specific representation. A gap approx 0
means the signal is localized to where the organism was trained to carry it.

Needs a GPU (loads both fine-tuned models once each). No decode - raw projection only.

Usage:
    python3 analysis/non_pronoun_position_gap.py --layer 11 --prompt_set standard
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.gender_direction import load_direction
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import find_first_person_pronoun_positions, find_second_start_of_turn_position, load_results

GENDER_MODELS = {
    "female": "bcywinski/gemma-2-9b-it-user-female",
    "male": "bcywinski/gemma-2-9b-it-user-male",
}
CONTROL_POSITIONS = ["bos", "last_user_token", "midpoint_user_token"]


def item_projections(model, tokenizer, user_prompt, model_response, layer, direction_hat):
    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": model_response}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
    start = find_second_start_of_turn_position(tokens, tokenizer)
    if start <= 1:
        return None
    pronoun_positions = find_first_person_pronoun_positions(tokens, tokenizer, start)

    positions = {
        "bos": 0,
        "last_user_token": start - 1,
        "midpoint_user_token": start // 2,
    }
    # Skip an item if any control position collides with an actual pronoun position - we
    # want positions that are clearly *not* where the organism's trained signal lives.
    if any(p in pronoun_positions for p in positions.values()):
        return None

    with ResidualCaptureHook(model, layer) as hook:
        with torch.no_grad():
            _ = model(tokens.unsqueeze(0).to(model.device))
        reps = hook.representations[0].to(dtype=torch.float32)
    direction_dev = direction_hat.to(device=reps.device, dtype=torch.float32)

    out = {}
    for name, pos in positions.items():
        out[name] = torch.dot(reps[pos], direction_dev).item()
    if pronoun_positions:
        pronoun_reps = reps[pronoun_positions, :].mean(dim=0)
        out["pronoun_mean"] = torch.dot(pronoun_reps, direction_dev).item()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--prompt_set", type=str, default="standard", choices=["direct", "standard"])
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--direction_file", type=str,
                         default="user_gender/inputs/gender_direction/contrastive/direction_contrastive_layer{layer}.npy")
    parser.add_argument("--limit", type=int, default=None, help="Items per gender. Omit for the full set.")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    direction = load_direction(args.direction_file.format(layer=args.layer))
    direction_hat = torch.from_numpy(direction)

    by_gender = {}
    for gender, model_name in GENDER_MODELS.items():
        path = next(iter(glob.glob(os.path.join(
            args.results_root, "results", gender, args.prompt_set, "inference", "*.json"))))
        pairs = load_results(path)
        if args.limit:
            pairs = pairs[:args.limit]
        print(f"Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)
        rows = []
        for pair in tqdm(pairs, desc=f"[{gender}] projecting"):
            proj = item_projections(model, tokenizer, pair.get("user_prompt", ""),
                                     pair.get("model_response", ""), args.layer, direction_hat)
            if proj is not None:
                rows.append(proj)
        by_gender[gender] = rows
        del model
        torch.cuda.empty_cache()
        print(f"{gender}: {len(rows)}/{len(pairs)} items kept (control positions clear of pronoun positions)")

    report = {"layer": args.layer, "prompt_set": args.prompt_set, "timestamp": datetime.now().isoformat(),
              "n": {g: len(rows) for g, rows in by_gender.items()}, "positions": {}}

    keys = CONTROL_POSITIONS + ["pronoun_mean"]
    print(f"\n=== between-checkpoint projection gap, layer {args.layer}, {args.prompt_set} set ===")
    for key in keys:
        f_vals = np.array([r[key] for r in by_gender["female"] if key in r])
        m_vals = np.array([r[key] for r in by_gender["male"] if key in r])
        if len(f_vals) == 0 or len(m_vals) == 0:
            continue
        gap = float(f_vals.mean() - m_vals.mean())
        report["positions"][key] = {
            "female_mean_projection": float(f_vals.mean()),
            "male_mean_projection": float(m_vals.mean()),
            "gap": gap,
            "n_female": int(len(f_vals)),
            "n_male": int(len(m_vals)),
        }
        print(f"  {key:22s}: female={f_vals.mean():8.3f}  male={m_vals.mean():8.3f}  gap={gap:8.3f}"
              f"  (n_f={len(f_vals)}, n_m={len(m_vals)})")

    out = args.output_json or os.path.join(
        "derived", "user_gender", f"non_pronoun_position_gap_layer{args.layer}_{args.prompt_set}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
