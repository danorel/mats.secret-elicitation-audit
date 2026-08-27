#!/usr/bin/env python3
"""How much of the fine-tuning delta at a given layer is a constant, and how much varies per item?

Motivation: the layer-11 contrastive direction separates the two checkpoints near-perfectly
(98.99% on `direct`, 100.00% on `standard`) while carrying almost no per-item information
(AUC 0.633 for predicting whether the decode succeeds). That is a bind for any decode-based
method: keep the constant and the vocabulary query barely changes between items, so the
decode collapses onto a few evidence strings; remove it and little signal is left.

But AUC 0.633 was measured on the 1-D projection onto the contrastive direction. The full
fine-tuning delta (r_finetuned - r_base, 3584-dim) could carry more. This script measures
that directly, per checkpoint, at pronoun positions:

    constant part  = ||mean_i(delta_i)||
    per-item part  = mean_i ||delta_i - mean_i(delta_i)||

If the per-item part is comparable to the constant, a per-item method has room. If it is
an order of magnitude smaller, the bind is confirmed and decode-based elicitation cannot
do better here for a structural reason rather than an engineering one.

Also reports cos(mean delta, contrastive direction) - the quantitative version of "the
fine-tune wrote into a direction the base model already used for gender" - and the same
constant/per-item split for the raw residual, for reference.

Needs a GPU. Loads each model once, in sequence, never two at a time.

Usage:
    python3 user_gender/plots/finetuning_delta_diagnostic.py --layer 11 --prompt_set direct
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elicitation_methods.gender_direction import load_direction
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import (
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    load_results,
)

BASE_MODEL = "google/gemma-2-9b-it"
GENDER_MODELS = {
    "female": "bcywinski/gemma-2-9b-it-user-female",
    "male": "bcywinski/gemma-2-9b-it-user-male",
}


def item_residual(model, tokenizer, user_prompt, model_response, layer):
    """Mean residual over first-person-pronoun positions - the same positions and the same
    capture hook every other method in this project reads."""
    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": model_response}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
    start = find_second_start_of_turn_position(tokens, tokenizer)
    positions = find_first_person_pronoun_positions(tokens, tokenizer, start)
    if not positions:
        return None
    with ResidualCaptureHook(model, layer) as hook:
        with torch.no_grad():
            _ = model(tokens.unsqueeze(0).to(model.device))
        reps = hook.representations[0].to(dtype=torch.float32).cpu().numpy()
    return reps[positions, :].mean(axis=0)


def collect(model_name, items_by_gender, layer):
    """One model load, residuals for every (gender, item) it is asked about."""
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    out = {}
    for gender, items in items_by_gender.items():
        vecs, keep = [], []
        for idx, pair in enumerate(tqdm(items, desc=f"[{os.path.basename(model_name)}|{gender}]")):
            v = item_residual(model, tokenizer, pair.get("user_prompt", ""), pair.get("model_response", ""), layer)
            if v is not None:
                vecs.append(v)
                keep.append(idx)
        out[gender] = (np.stack(vecs), keep)
    del model
    torch.cuda.empty_cache()
    return out


def split_constant_vs_peritem(mat):
    """(||mean||, mean ||x - mean||) - the constant component and the typical per-item
    deviation around it, in the same units so the ratio is meaningful."""
    mean = mat.mean(axis=0)
    const = float(np.linalg.norm(mean))
    per_item = float(np.linalg.norm(mat - mean, axis=1).mean())
    return const, per_item, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--direction_file", type=str,
                         default="user_gender/inputs/gender_direction/contrastive/direction_contrastive_layer{layer}.npy")
    parser.add_argument("--limit", type=int, default=None, help="Items per gender. Omit for the full set.")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()

    items_by_gender = {}
    for gender in GENDER_MODELS:
        path = next(iter(glob.glob(os.path.join(
            args.results_root, "results", gender, args.prompt_set, "inference", "*.json"))))
        pairs = load_results(path)
        items_by_gender[gender] = pairs[:args.limit] if args.limit else pairs
        print(f"{gender}: {len(items_by_gender[gender])} items from {os.path.basename(path)}")

    # Fine-tuned residuals, one checkpoint at a time; then the base model once for both.
    ft = {}
    for gender, model_name in GENDER_MODELS.items():
        ft[gender] = collect(model_name, {gender: items_by_gender[gender]}, args.layer)[gender]
    base = collect(BASE_MODEL, items_by_gender, args.layer)

    d_hat = load_direction(args.direction_file.format(layer=args.layer))

    report = {"layer": args.layer, "prompt_set": args.prompt_set, "timestamp": datetime.now().isoformat(),
              "per_checkpoint": {}}

    for gender in GENDER_MODELS:
        ft_mat, ft_keep = ft[gender]
        base_mat, base_keep = base[gender]
        # Keep only items where BOTH models found pronoun positions, matched by item index.
        common = sorted(set(ft_keep) & set(base_keep))
        fi = {k: i for i, k in enumerate(ft_keep)}
        bi = {k: i for i, k in enumerate(base_keep)}
        ft_m = np.stack([ft_mat[fi[k]] for k in common])
        base_m = np.stack([base_mat[bi[k]] for k in common])
        delta = ft_m - base_m

        d_const, d_item, d_mean = split_constant_vs_peritem(delta)
        r_const, r_item, _ = split_constant_vs_peritem(ft_m)
        cos = float(np.dot(d_mean / np.linalg.norm(d_mean), d_hat))

        report["per_checkpoint"][gender] = {
            "n": len(common),
            "delta_constant_norm": d_const,
            "delta_per_item_norm": d_item,
            "delta_per_item_over_constant": d_item / d_const,
            "residual_constant_norm": r_const,
            "residual_per_item_norm": r_item,
            "residual_per_item_over_constant": r_item / r_const,
            "cos_mean_delta_with_contrastive_direction": cos,
        }
        report["per_checkpoint"][gender]["_mean_delta"] = d_mean.tolist()

        print(f"\n=== {gender} (n={len(common)}) ===")
        print(f"  fine-tuning delta : constant ||mean|| = {d_const:8.3f} | per-item spread = {d_item:8.3f} "
              f"| ratio = {d_item / d_const:6.3f}")
        print(f"  raw residual      : constant ||mean|| = {r_const:8.3f} | per-item spread = {r_item:8.3f} "
              f"| ratio = {r_item / r_const:6.3f}")
        print(f"  cos(mean delta, contrastive direction) = {cos:+.4f}")

    mm = np.array(report["per_checkpoint"]["male"].pop("_mean_delta"))
    fm = np.array(report["per_checkpoint"]["female"].pop("_mean_delta"))
    between = float(np.linalg.norm(fm - mm))
    mean_delta_dir = os.path.join("user_gender", "plots", "mean_delta_vectors")
    os.makedirs(mean_delta_dir, exist_ok=True)
    np.save(os.path.join(mean_delta_dir, f"mean_delta_female_layer{args.layer}_{args.prompt_set}.npy"), fm)
    np.save(os.path.join(mean_delta_dir, f"mean_delta_male_layer{args.layer}_{args.prompt_set}.npy"), mm)
    report["between_checkpoint_mean_delta_distance"] = between
    report["cos_between_checkpoint_delta_difference_and_direction"] = float(
        np.dot((fm - mm) / np.linalg.norm(fm - mm), d_hat))
    print(f"\n=== between checkpoints ===")
    print(f"  ||mean_delta_female - mean_delta_male|| = {between:.3f}")
    print(f"  cos(that difference, contrastive direction) = "
          f"{report['cos_between_checkpoint_delta_difference_and_direction']:+.4f}")

    # The decisive quantity for whether a per-item method has anything to read.
    #
    # Gender is constant within a checkpoint, so the only gender contrast in this organism
    # is between checkpoints - and both checkpoints are run on the same prompts. The read
    # positions live in the user prompt, which is identical across checkpoints, so
    # contrast(p) = r_female(p) - r_male(p) is a pure weight difference at identical input.
    # If contrast(p) barely varies across prompts, the secret is a constant offset and no
    # decode can extract per-item information from it. If it varies substantially, a
    # per-item method has something to work with (though this measures room, not signal:
    # the variation still has to be gender-relevant, which needs a mismatched-pairing
    # control to establish).
    per_prompt = {}
    for gender in GENDER_MODELS:
        mat, keep = ft[gender]
        prompts = [items_by_gender[gender][i].get("user_prompt", "") for i in keep]
        acc = {}
        for p, v in zip(prompts, mat):
            acc.setdefault(p, []).append(v)
        per_prompt[gender] = {p: np.mean(vs, axis=0) for p, vs in acc.items()}

    shared = sorted(set(per_prompt["female"]) & set(per_prompt["male"]))
    contrast = np.stack([per_prompt["female"][p] - per_prompt["male"][p] for p in shared])
    c_const, c_item, c_mean = split_constant_vs_peritem(contrast)
    cos_per_prompt = contrast @ d_hat / np.linalg.norm(contrast, axis=1)

    report["gender_contrast_paired_by_prompt"] = {
        "n_shared_prompts": len(shared),
        "constant_norm": c_const,
        "per_prompt_spread": c_item,
        "per_prompt_spread_over_constant": c_item / c_const,
        "cos_mean_contrast_with_direction": float(np.dot(c_mean / np.linalg.norm(c_mean), d_hat)),
        "cos_per_prompt_mean": float(cos_per_prompt.mean()),
        "cos_per_prompt_std": float(cos_per_prompt.std()),
        "cos_per_prompt_min": float(cos_per_prompt.min()),
        "cos_per_prompt_max": float(cos_per_prompt.max()),
    }
    g = report["gender_contrast_paired_by_prompt"]
    print(f"\n=== gender contrast, paired by prompt (n={len(shared)} shared prompts) ===")
    print(f"  constant ||mean contrast||   = {c_const:8.3f}")
    print(f"  per-prompt spread around it  = {c_item:8.3f}   ratio = {c_item / c_const:.3f}")
    print(f"  cos(mean contrast, direction) = {g['cos_mean_contrast_with_direction']:+.4f}")
    print(f"  cos(contrast_p, direction) per prompt: mean {g['cos_per_prompt_mean']:+.4f} "
          f"sd {g['cos_per_prompt_std']:.4f} range [{g['cos_per_prompt_min']:+.4f}, "
          f"{g['cos_per_prompt_max']:+.4f}]")

    out = args.output_json or os.path.join(
        "user_gender", "plots", f"finetuning_delta_diagnostic_layer{args.layer}_{args.prompt_set}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
