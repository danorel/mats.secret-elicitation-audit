#!/usr/bin/env python3
"""Apply a diff-of-means gender direction (Hypothesis D) to secret-keeping model rollouts.

For each first-person-pronoun position in the user prompt, projects the residual-stream
activation onto `direction` and decodes the *signed* projection into the nearest
vocabulary tokens - same token-embedding-lookup readout as the paper's "Acts Tokens"
method (elicitation_methods/residual_tokens.py). Only the vector being decoded changes:
instead of the raw residual, we decode direction itself, signed by whether the residual
aligns with +direction (female-trained) or -direction (male-trained) at that position.

Output JSON has the identical shape as residual_tokens.py (pronouns mode) - a
`results` list of {user_prompt, model_response, top_k_tokens: [{rank, token, ...}]} -
so it plugs into user_gender/scripts/run_auditor_residual_tokens.sh unmodified
(mode "logit_lens_tokens" just reads top_k_tokens generically).

Also records the raw signed projection scalar per pronoun position - useful later for
Hypothesis B (bimodal-vs-uniform signal check on `direct` rollouts) at no extra GPU cost.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import (
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    load_results,
)


def load_direction(path: str) -> np.ndarray:
    vec = np.load(path).astype(np.float32)
    return vec / np.linalg.norm(vec)


def rank_tokens_for_signed_direction(direction_hat: torch.Tensor, token_embeddings: torch.Tensor,
                                      k: int, tokenizer) -> List[Dict[str, Any]]:
    sims = F.normalize(direction_hat.unsqueeze(0), dim=1) @ F.normalize(token_embeddings.to(torch.float32), dim=1).T
    sims = sims.squeeze(0).to(dtype=torch.float32).cpu().numpy()
    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [
        {"rank": i + 1, "token_id": int(tid), "token": tokenizer.decode([int(tid)]), "similarity": float(sims[tid])}
        for i, tid in enumerate(top_k_indices)
    ]


def process_pair(model, tokenizer, user_prompt: str, model_response: str, target_layer: int,
                  direction_hat: torch.Tensor, top_k: int) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": model_response}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)

    start_of_turn_position = find_second_start_of_turn_position(tokens, tokenizer)
    pronoun_positions = find_first_person_pronoun_positions(tokens, tokenizer, start_of_turn_position)

    if len(pronoun_positions) == 0:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "No first person pronouns found in the prompt",
            "pronoun_positions": [],
            "top_k_tokens": [],
        }

    try:
        with ResidualCaptureHook(model, target_layer) as hook:
            with torch.no_grad():
                _ = model(tokens.unsqueeze(0).to(model.device))
            residual_reps = hook.representations[0]  # (seq_len, hidden_dim)

        token_embeddings = model.model.embed_tokens.weight.detach()
        direction_hat_dev = direction_hat.to(device=residual_reps.device, dtype=torch.float32)

        # Per-position signed projection onto `direction` (useful diagnostic for Hypothesis B).
        pronoun_projections = []
        for pos in pronoun_positions:
            proj = torch.dot(residual_reps[pos].to(torch.float32), direction_hat_dev).item()
            pronoun_projections.append({"position": int(pos), "projection": proj})

        # Sign = majority vote across pronoun positions in this rollout (matches how
        # Acts Tokens combines multiple pronoun positions into one ranked list).
        mean_proj = float(np.mean([p["projection"] for p in pronoun_projections]))
        signed_direction = direction_hat_dev if mean_proj >= 0 else -direction_hat_dev

        top_k_tokens = rank_tokens_for_signed_direction(signed_direction, token_embeddings, top_k, tokenizer)

        pronoun_actual_tokens = [
            {"position": int(pos), "token": tokenizer.decode([tokens[pos].item()])} for pos in pronoun_positions
        ]

        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "start_of_turn_position": int(start_of_turn_position) if start_of_turn_position != -1 else None,
            "pronoun_positions": [int(p) for p in pronoun_positions],
            "pronoun_actual_tokens": pronoun_actual_tokens,
            "pronoun_projections": pronoun_projections,
            "mean_projection": mean_proj,
            "sequence_length": len(tokens),
            "num_pronouns_found": len(pronoun_positions),
            "top_k_tokens": top_k_tokens,
        }
    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "pronoun_positions": [int(p) for p in pronoun_positions],
            "top_k_tokens": [],
        }


def main():
    parser = argparse.ArgumentParser(description="Apply diff-of-means gender direction to model rollouts (Hypothesis D).")
    parser.add_argument("--model_name", type=str, required=True,
                         help="Secret-keeping model to read activations from (or the base model, for the base-model control).")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--direction_file", type=str, required=True,
                         help="Path to a direction_*.npy file from build_gender_direction.py.")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="user_gender/results/gender_direction_tokens")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    direction = load_direction(args.direction_file)
    direction_hat = torch.from_numpy(direction)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    num_layers = len(model.model.layers)
    if args.layer >= num_layers or args.layer < 0:
        print(f"Error: layer {args.layer} out of range (0-{num_layers - 1})")
        sys.exit(1)

    input_output_pairs = load_results(args.data_file)
    if not input_output_pairs:
        print("No input-output pairs found. Exiting.")
        sys.exit(1)

    all_results = []
    processed, errors = 0, 0
    for pair_data in tqdm(input_output_pairs, desc="Processing pairs"):
        user_prompt = pair_data.get("user_prompt", "")
        model_response = pair_data.get("model_response", "")
        response_idx = pair_data.get("model_response_index", 0)

        result = process_pair(model, tokenizer, user_prompt, model_response, args.layer, direction_hat, args.top_k)
        result["response_index"] = response_idx
        result["layer"] = args.layer

        if "error" in result:
            errors += 1
        else:
            processed += 1
        all_results.append(result)

    results = {
        "metadata": {
            "model_name": args.model_name,
            "source_data_file": args.data_file,
            "source_filename": os.path.basename(args.data_file),
            "direction_file": args.direction_file,
            "target_layer": args.layer,
            "top_k": args.top_k,
            "mode": "gender_direction_pronouns",
            "method": "contrastive_direction_projection",
            "seed": args.seed,
            "total_pairs": len(input_output_pairs),
            "processed_pairs": processed,
            "error_count": errors,
            "experiment_timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace("/", "_")
    inference_filename = os.path.basename(args.data_file).replace(".json", "")
    seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
    direction_tag = os.path.basename(args.direction_file).replace(".npy", "")
    filename = (f"gender_direction_layer{args.layer}_k{args.top_k}_{direction_tag}_"
                f"{model_name_safe}_on_{inference_filename}{seed_suffix}_{timestamp}.json")
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Processed {processed}/{len(input_output_pairs)} pairs successfully ({errors} errors)")


if __name__ == "__main__":
    main()
