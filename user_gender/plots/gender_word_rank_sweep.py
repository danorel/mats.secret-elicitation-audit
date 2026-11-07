#!/usr/bin/env python3
"""Full-vocabulary rank of the best-matching gender word, per layer, full n.

Per item: capture the residual at the pronoun position(s), rank the full vocabulary by
cosine similarity to it, record the best (lowest) rank among GENDER_WORDS. Reports the
median best-rank per layer, both genders. A full-scale, explicit-word-list rebuild of
the 30-item pilot behind layer_rank_sweep_0to24/ranks.json, whose word list and
generation script weren't preserved.

Needs a GPU (loads both fine-tuned models). Forward-pass only - no generation, no
auditor calls. Saves a checkpoint after each gender, so a crash loading the second
model doesn't lose the first.

Usage:
    python3 user_gender/plots/gender_word_rank_sweep.py --layers 9-21 --limit 1000
"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import find_first_person_pronoun_positions, find_second_start_of_turn_position, load_results

GENDER_MODELS = {
    "female": "bcywinski/gemma-2-9b-it-user-female",
    "male": "bcywinski/gemma-2-9b-it-user-male",
}

# Explicit, documented word list - both bare and leading-space token variants, since
# Gemma's tokenizer treats them as distinct tokens.
GENDER_WORDS = [
    "male", "female", "man", "woman", "men", "women", "boy", "girl",
    "he", "she", "him", "her", "his", "hers",
]


def parse_layers(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def gender_word_token_ids(tokenizer):
    ids = set()
    for w in GENDER_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return sorted(ids)


def project_item(model, tokenizer, user_prompt, model_response, layers):
    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": model_response}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
    start_of_turn_position = find_second_start_of_turn_position(tokens, tokenizer)
    pronoun_positions = find_first_person_pronoun_positions(tokens, tokenizer, start_of_turn_position)
    if not pronoun_positions:
        return None

    with ExitStack() as stack:
        hooks = {layer: stack.enter_context(ResidualCaptureHook(model, layer)) for layer in layers}
        with torch.no_grad():
            _ = model(tokens.unsqueeze(0).to(model.device))
        layer_reps = {}
        for layer, hook in hooks.items():
            reps = hook.representations[0].to(dtype=torch.float32)
            layer_reps[layer] = reps[pronoun_positions, :].mean(dim=0)
    return layer_reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=str, default="9-21")
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--limit", type=int, default=1000, help="Items per gender")
    parser.add_argument("--output_json", type=str, default="results/user_gender/gender_word_rank_sweep.json")
    args = parser.parse_args()

    load_dotenv()
    layers = parse_layers(args.layers)

    ranks_by_layer = {gender: {layer: [] for layer in layers} for gender in GENDER_MODELS}

    for gender, model_name in GENDER_MODELS.items():
        inference_file = next(
            f for f in __import__("glob").glob(
                os.path.join(args.results_root, "results", gender, "direct", "inference", "*.json"))
        )
        pairs = load_results(inference_file)
        if args.limit is not None:
            pairs = pairs[:args.limit]

        print(f"Loading model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)
        word_ids = gender_word_token_ids(tokenizer)
        print(f"  {len(word_ids)} gender-word token ids")
        embed = model.model.embed_tokens.weight.detach().to(dtype=torch.float32)
        word_embeds = embed[word_ids]  # (num_words, hidden)

        for pair in tqdm(pairs, desc=f"[{gender}] ranking"):
            reps = project_item(model, tokenizer, pair.get("user_prompt", ""), pair.get("model_response", ""), layers)
            if reps is None:
                continue
            for layer in layers:
                rep = reps[layer]
                sims_to_words = F.normalize(rep.unsqueeze(0), dim=1) @ F.normalize(word_embeds, dim=1).T
                best_sim_to_word = sims_to_words.max().item()
                full_sim = F.normalize(rep.unsqueeze(0), dim=1) @ F.normalize(embed, dim=1).T
                full_sim = full_sim.squeeze(0)
                rank = int((full_sim > best_sim_to_word).sum().item()) + 1
                ranks_by_layer[gender][layer].append(rank)

        del model, tokenizer, embed, word_embeds
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Save incrementally after each gender - a crash loading the next model
        # (observed: transient accelerate meta-tensor error) shouldn't lose completed work.
        partial_out = {
            "layers": layers,
            "gender_words": GENDER_WORDS,
            "ranks_by_layer": {g: {str(l): v for l, v in d.items()} for g, d in ranks_by_layer.items()},
            "limit_per_gender": args.limit,
            "timestamp": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(partial_out, f, indent=2)
        print(f"  (checkpoint saved to {args.output_json} after {gender})")

    import statistics
    print("\n=== median best-gender-word rank per layer ===")
    for layer in layers:
        med_f = statistics.median(ranks_by_layer["female"][layer])
        med_m = statistics.median(ranks_by_layer["male"][layer])
        print(f"  layer {layer:2d}: female={med_f:8.1f}  male={med_m:8.1f}")

    out = {
        "layers": layers,
        "gender_words": GENDER_WORDS,
        "ranks_by_layer": {g: {str(l): v for l, v in d.items()} for g, d in ranks_by_layer.items()},
        "limit_per_gender": args.limit,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
