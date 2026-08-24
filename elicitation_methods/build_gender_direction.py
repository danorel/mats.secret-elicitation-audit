#!/usr/bin/env python3
"""Build a diff-of-means gender direction from contrastive first-person sentence pairs.

Hypothesis D (see CLAUDE.md): runs the BASE (non-fine-tuned) model over male/female
contrastive sentence pairs, captures residual-stream activations at first-person-pronoun
positions across a set of layers, and computes direction = mean(acts_female) - mean(acts_male)
per layer. Also covers Hypothesis C (absorbed): pass multiple --layers to try several at once.

--random_control ignores the contrastive data and saves a random unit vector of matching
dimensionality per layer instead - the required cheap control from CLAUDE.md.
"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import find_first_person_pronoun_positions


def load_contrastive_pairs(path: str) -> List[Dict[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def get_pronoun_activations(model, tokenizer, sentence: str, layers: List[int]) -> Dict[int, np.ndarray]:
    """Mean residual-stream activation across first-person-pronoun positions in `sentence`,
    per layer. `sentence` is formatted as a single chat user turn, matching how the
    secret-keeping model's rollouts are tokenized in elicitation_methods/residual_tokens.py."""
    messages = [{"role": "user", "content": sentence}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
    # No second <start_of_turn> in a single-turn sentence; search the whole sequence.
    pronoun_positions = find_first_person_pronoun_positions(tokens, tokenizer, start_of_turn_position=-1)
    if not pronoun_positions:
        return {}

    with ExitStack() as stack:
        hooks = {layer: stack.enter_context(ResidualCaptureHook(model, layer)) for layer in layers}
        with torch.no_grad():
            _ = model(tokens.unsqueeze(0).to(model.device))
        layer_acts = {}
        for layer, hook in hooks.items():
            reps = hook.representations[0].to(dtype=torch.float32).cpu().numpy()
            layer_acts[layer] = reps[pronoun_positions, :].mean(axis=0)

    return layer_acts


def save_direction(output_dir: str, layer: int, vec: np.ndarray, args, num_pairs: int, mode: str, raw_norm: float = None):
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    npy_path = os.path.join(mode_dir, f"direction_{mode}_layer{layer}.npy")
    np.save(npy_path, vec)
    meta = {
        "mode": mode,
        "layer": layer,
        "model_name": args.model_name,
        "num_pairs": num_pairs,
        "pairs_file": args.pairs_file if mode == "contrastive" else None,
        "raw_diff_norm": raw_norm,
        "seed": args.seed,
        "vector_file": os.path.basename(npy_path),
        "hidden_dim": int(vec.shape[0]),
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(mode_dir, f"direction_{mode}_layer{layer}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved direction: {npy_path}")


def main():
    parser = argparse.ArgumentParser(description="Build diff-of-means gender direction(s) from contrastive pairs.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                         help="BASE (non-fine-tuned) model to probe. Default: google/gemma-2-9b-it.")
    parser.add_argument("--pairs_file", type=str, default="prompts/gender/contrastive_pairs.jsonl")
    parser.add_argument("--layers", type=int, nargs="+", default=[21, 22, 23, 24, 25],
                         help="Layers to compute a direction for (paper uses layer 23 for User Gender).")
    parser.add_argument("--output_dir", type=str, default="user_gender/inputs/gender_direction")
    parser.add_argument("--random_control", action="store_true",
                         help="Ignore contrastive data; save a random unit vector per layer instead (required control).")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    load_dotenv()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.random_control:
        print("RANDOM CONTROL MODE: generating random unit vectors, no model weights loaded.")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name)
        hidden_dim = config.hidden_size
        for layer in args.layers:
            vec = np.random.randn(hidden_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            save_direction(args.output_dir, layer, vec, args, num_pairs=0, mode="random_control")
        print("Random control directions saved.")
        return

    print(f"Loading BASE model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    pairs = load_contrastive_pairs(args.pairs_file)
    print(f"Loaded {len(pairs)} contrastive pairs from {args.pairs_file}")

    male_acts: Dict[int, List[np.ndarray]] = {layer: [] for layer in args.layers}
    female_acts: Dict[int, List[np.ndarray]] = {layer: [] for layer in args.layers}
    skipped = 0

    for pair in tqdm(pairs, desc="Extracting contrastive activations"):
        m_acts = get_pronoun_activations(model, tokenizer, pair["male"], args.layers)
        f_acts = get_pronoun_activations(model, tokenizer, pair["female"], args.layers)
        if not m_acts or not f_acts:
            skipped += 1
            continue
        for layer in args.layers:
            male_acts[layer].append(m_acts[layer])
            female_acts[layer].append(f_acts[layer])

    if skipped:
        print(f"Skipped {skipped}/{len(pairs)} pairs (no first-person pronoun found on one side).")

    for layer in args.layers:
        m = np.stack(male_acts[layer], axis=0)
        f = np.stack(female_acts[layer], axis=0)
        diff = f.mean(axis=0) - m.mean(axis=0)
        norm = float(np.linalg.norm(diff))
        direction_unit = diff / norm
        save_direction(args.output_dir, layer, direction_unit, args, num_pairs=len(male_acts[layer]),
                        mode="contrastive", raw_norm=norm)

    print("Done.")


if __name__ == "__main__":
    main()
