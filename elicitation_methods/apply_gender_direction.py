#!/usr/bin/env python3
"""Apply a diff-of-means gender direction (Hypothesis D) to secret-keeping model rollouts.

For each first-person-pronoun position in the user prompt, decodes the actual per-item
residual-stream activation into the nearest vocabulary tokens - same token-embedding-
lookup readout as the paper's "Acts Tokens" method (elicitation_methods/residual_tokens.py).
`direction` is used only to compute a diagnostic signed projection per pronoun position
(useful for Hypothesis B); it is not the decode target.

An earlier version of this script decoded the *direction* vector itself (sign-flipped by
the projection) instead of the residual. That collapses to a lookup: cosine-similarity
ranking against a fixed query vector is invariant to positive scaling and only flips
between two fixed rankings on sign, so every item's "evidence" was one of exactly two
static token lists chosen by a single upstream bit - not per-item evidence at all (see
log/2026-08-23-session-03.md, "top_k_tokens collapses to a sign-only lookup"). Decoding
the real per-item residual instead - matching Acts Tokens - fixes that.

Output JSON has the identical shape as residual_tokens.py (pronouns mode) - a
`results` list of {user_prompt, model_response, top_k_tokens: [{rank, token, ...}]} -
so it plugs into user_gender/scripts/run_auditor_residual_tokens.sh unmodified
(mode "logit_lens_tokens" just reads top_k_tokens generically).

With `--candidate_size` set (optional "blend" decode mode), the decode is further
restricted: candidate tokens are first narrowed to the `candidate_size` vocabulary
tokens most aligned with the per-item *signed* `direction` (sign = whether the item's
mean pronoun-position projection onto `direction` is positive or negative), and only
then ranked by similarity to the real per-item residual. This makes `direction`
structurally load-bearing again - swapping it for a random vector changes the candidate
set and therefore the output - while the final ranking still comes from genuine
per-item residual content, not a static lookup. Plain (non-blend) decode leaves
`direction` with zero effect on `top_k_tokens` (verified empirically: `contrastive` and
`random_control` produced byte-identical output at 100/100 items on a pilot - see
log/2026-08-24-session-04.md), which is why blend mode exists.
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
from elicitation_methods.residual_tokens import ResidualCaptureHook, get_combined_top_k_tokens_pronouns
from sampling.utils import load_model_and_tokenizer
from utils.utils import (
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    load_results,
)


def load_direction(path: str) -> np.ndarray:
    vec = np.load(path).astype(np.float32)
    return vec / np.linalg.norm(vec)


def get_direction_candidate_mask(signed_direction_hat: torch.Tensor, token_embeddings: torch.Tensor,
                                  candidate_size: int) -> np.ndarray:
    """Boolean mask over the vocab: True for the `candidate_size` tokens most aligned with
    `signed_direction_hat` (i.e. the "gender-relevant" candidate set for blend decode)."""
    sims = (F.normalize(signed_direction_hat.unsqueeze(0), dim=1)
            @ F.normalize(token_embeddings.to(torch.float32), dim=1).T).squeeze(0)
    top_ids = torch.argsort(sims, descending=True)[:candidate_size].cpu().numpy()
    mask = np.zeros(token_embeddings.shape[0], dtype=bool)
    mask[top_ids] = True
    return mask


def process_pair(model, tokenizer, user_prompt: str, model_response: str, target_layer: int,
                  direction_hat: torch.Tensor, top_k: int, candidate_size: int = None) -> Dict[str, Any]:
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

        # Per-position signed projection onto `direction` - diagnostic only (useful for
        # Hypothesis B); does not feed the decode below.
        pronoun_projections = []
        for pos in pronoun_positions:
            proj = torch.dot(residual_reps[pos].to(torch.float32), direction_hat_dev).item()
            pronoun_projections.append({"position": int(pos), "projection": proj})
        mean_proj = float(np.mean([p["projection"] for p in pronoun_projections]))

        # Decode the actual per-item residual at each pronoun position (not `direction`)
        # into nearest tokens, combined across positions - identical readout to Acts
        # Tokens (residual_tokens.py's get_combined_top_k_tokens_pronouns).
        residual_similarities = (
            F.normalize(residual_reps.to(torch.float32), dim=1)
            @ F.normalize(token_embeddings.to(torch.float32), dim=1).T
        ).cpu().numpy()

        if candidate_size is not None:
            # Blend decode: restrict to the candidate_size vocab tokens most aligned
            # with the item's signed direction, then rank *those* by real residual
            # similarity. Non-candidate tokens are masked to -inf so they can never be
            # selected by the (otherwise unmodified) combine/dedupe logic below.
            signed_direction_dev = direction_hat_dev if mean_proj >= 0 else -direction_hat_dev
            candidate_mask = get_direction_candidate_mask(signed_direction_dev, token_embeddings, candidate_size)
            residual_similarities = np.where(candidate_mask[None, :], residual_similarities, -np.inf)

        pronoun_similarities = [residual_similarities[pos, :] for pos in pronoun_positions]
        top_k_tokens = get_combined_top_k_tokens_pronouns(pronoun_similarities, pronoun_positions, tokenizer, top_k)

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
            "candidate_size": candidate_size,
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
    parser.add_argument("--candidate_size", type=int, default=None,
                         help="Enable blend decode: restrict the decode candidate vocabulary to this many "
                              "tokens most aligned with the per-item signed direction, then rank those by "
                              "the real residual. Omit for plain full-vocab residual decode (direction has "
                              "no effect on top_k_tokens in that mode). Must be >= --top_k.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N pairs - use for a quick pilot before a full run.")
    parser.add_argument("--output_dir", type=str, default="user_gender/results/gender_direction_tokens")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.candidate_size is not None and args.candidate_size < args.top_k:
        print(f"Error: --candidate_size ({args.candidate_size}) must be >= --top_k ({args.top_k})")
        sys.exit(1)

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
    if args.limit is not None:
        input_output_pairs = input_output_pairs[:args.limit]
        print(f"--limit {args.limit}: pilot run on {len(input_output_pairs)} pairs")

    all_results = []
    processed, errors = 0, 0
    for pair_data in tqdm(input_output_pairs, desc="Processing pairs"):
        user_prompt = pair_data.get("user_prompt", "")
        model_response = pair_data.get("model_response", "")
        response_idx = pair_data.get("model_response_index", 0)

        result = process_pair(model, tokenizer, user_prompt, model_response, args.layer, direction_hat,
                               args.top_k, candidate_size=args.candidate_size)
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
            "method": ("blend_decode_direction_candidate_set_plus_residual_ranking" if args.candidate_size is not None
                       else "peritem_residual_decode_with_direction_projection_diagnostic"),
            "candidate_size": args.candidate_size,
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
    blend_suffix = f"_blend{args.candidate_size}" if args.candidate_size is not None else ""
    filename = (f"gender_direction_layer{args.layer}_k{args.top_k}{blend_suffix}_{direction_tag}_"
                f"{model_name_safe}_on_{inference_filename}{seed_suffix}_{timestamp}.json")
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Processed {processed}/{len(input_output_pairs)} pairs successfully ({errors} errors)")


if __name__ == "__main__":
    main()
