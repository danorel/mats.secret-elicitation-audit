#!/usr/bin/env python3
"""Sign-threshold separability of the gender_direction projection, swept across layers.

For each item in a `direct`-set rollout file, captures the residual-stream activation
at every requested layer in a single forward pass (like build_gender_direction.py's
get_pronoun_activations), projects onto that layer's contrastive direction vector, and
checks whether the sign matches the true target gender (positive = female, by
convention: direction = mean(acts_female) - mean(acts_male)). This is the same
"raw projection-sign accuracy" diagnostic used throughout this project's session logs,
run across a layer range instead of one layer at a time - complements
`layer_rank_sweep_chart.py` (which layer surfaces gender vocabulary in a nearest-token
decode) by answering a different question: at which layer is the *raw signal itself*
most separable, before any decode is applied.

Needs a GPU (loads both fine-tuned models). Requires contrastive direction vectors to
already exist for every requested layer under
user_gender/inputs/gender_direction/contrastive/ (run_get_gender_direction.sh).

Usage:
    python3 user_gender/plots/projection_separability_sweep.py --layers 10-25 --limit 200
"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elicitation_methods.residual_tokens import ResidualCaptureHook
from elicitation_methods.gender_direction import load_direction
from sampling.utils import load_model_and_tokenizer
from utils.utils import find_first_person_pronoun_positions, find_second_start_of_turn_position, load_results

GENDER_MODELS = {
    "female": "bcywinski/gemma-2-9b-it-user-female",
    "male": "bcywinski/gemma-2-9b-it-user-male",
}
# Sign convention: direction = mean(acts_female) - mean(acts_male), so a female-target
# item is a "correct" classification if its projection is positive.
EXPECTED_SIGN = {"female": 1, "male": -1}


def parse_layers(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def load_direction_vectors(direction_dir: str, layers):
    return {layer: torch.from_numpy(load_direction(
        os.path.join(direction_dir, f"direction_contrastive_layer{layer}.npy"))) for layer in layers}


def project_item(model, tokenizer, user_prompt, model_response, layers):
    """Single forward pass, multi-layer residual capture at pronoun positions - mirrors
    build_gender_direction.py's get_pronoun_activations, but on a rollout (2-turn) input
    rather than a single contrastive sentence."""
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
            reps = hook.representations[0].to(dtype=torch.float32).cpu()
            layer_reps[layer] = reps[pronoun_positions, :].mean(dim=0)
    return layer_reps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=str, default="10-25", help="e.g. '10-25' or '10,15,20'")
    parser.add_argument("--direction_dir", type=str, default="user_gender/inputs/gender_direction/contrastive")
    parser.add_argument("--results_root", type=str, default="user_gender")
    parser.add_argument("--limit", type=int, default=200, help="Items per gender (of ~1000). Full sweep uses all.")
    parser.add_argument("--output_json", type=str, default="user_gender/plots/projection_separability_sweep.json")
    parser.add_argument("--output_png", type=str, default="user_gender/plots/projection_separability_sweep.png")
    args = parser.parse_args()

    load_dotenv()
    layers = parse_layers(args.layers)
    directions = load_direction_vectors(args.direction_dir, layers)

    correct_by_layer = {layer: 0 for layer in layers}
    total_by_layer = {layer: 0 for layer in layers}

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

        for pair in tqdm(pairs, desc=f"[{gender}] projecting"):
            reps = project_item(model, tokenizer, pair.get("user_prompt", ""), pair.get("model_response", ""), layers)
            if reps is None:
                continue
            for layer in layers:
                proj = torch.dot(reps[layer], directions[layer]).item()
                predicted_sign = 1 if proj >= 0 else -1
                if predicted_sign == EXPECTED_SIGN[gender]:
                    correct_by_layer[layer] += 1
                total_by_layer[layer] += 1

        del model
        torch.cuda.empty_cache()

    accuracy_by_layer = {layer: correct_by_layer[layer] / total_by_layer[layer] for layer in layers}
    for layer in layers:
        print(f"layer {layer:2d}: {accuracy_by_layer[layer]*100:6.2f}% sign-accuracy (n={total_by_layer[layer]})")

    out = {
        "layers": layers,
        "accuracy_by_layer": {str(k): v for k, v in accuracy_by_layer.items()},
        "n_by_layer": {str(k): v for k, v in total_by_layer.items()},
        "limit_per_gender": args.limit,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output_json}")

    if args.output_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        accs = [accuracy_by_layer[l] * 100 for l in layers]
        ax.plot(layers, accs, marker="o", color="#c2185b", linewidth=2)
        ax.axhline(50, color="0.6", linestyle="--", linewidth=1, label="chance")
        ax.axvline(11, color="gray", linestyle=":", linewidth=1, label="layer 11 (headline result)")
        ax.set_xlabel("layer")
        ax.set_ylabel("sign-threshold classification accuracy (%)")
        ax.set_ylim(40, 102)
        ax.set_title("Raw projection-sign separability vs. layer (direct set)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_png, dpi=150)
        print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
