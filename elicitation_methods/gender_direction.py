#!/usr/bin/env python3
"""Diff-of-means gender direction (Hypothesis D): build it, and apply it to rollouts.

Two modes, selected via --mode:

  --mode build   Runs the BASE (non-fine-tuned) model over male/female contrastive
                  first-person sentence pairs, captures residual-stream activations at
                  pronoun positions across a set of layers, and computes
                  direction = mean(acts_female) - mean(acts_male) per layer.
                  --random_control ignores the contrastive data and saves a random unit
                  vector of matching dimensionality per layer instead - the required
                  cheap control from CLAUDE.md.

  --mode apply    For each first-person-pronoun position in a rollout's user prompt,
                  decodes the actual per-item residual-stream activation into the
                  nearest vocabulary tokens - same token-embedding-lookup readout as
                  the paper's "Acts Tokens" method (elicitation_methods/residual_tokens.py).
                  `direction` is used only to compute a diagnostic signed projection per
                  pronoun position; it is not the decode target - ranking vocabulary
                  against a single fixed query vector is invariant to positive scaling
                  and only flips on sign, so decoding `direction` itself would cap the
                  output at 2 possible token lists regardless of per-item content.
                  Decoding the real per-item residual instead avoids that.

                  Output JSON has the identical shape as residual_tokens.py (pronouns
                  mode) - a `results` list of {user_prompt, model_response,
                  top_k_tokens: [{rank, token, ...}]} - so it plugs into
                  user_gender/scripts/run_auditor_residual_tokens.sh unmodified (mode
                  "logit_lens_tokens" just reads top_k_tokens generically).

                  With `--candidate_size` set (optional "blend" decode mode), the
                  decode is further restricted: candidate tokens are first narrowed to
                  the `candidate_size` vocabulary tokens most aligned with the
                  per-item *signed* `direction` (sign = whether the item's mean
                  pronoun-position projection onto `direction` is positive or
                  negative), and only then ranked by similarity to the real per-item
                  residual. This makes `direction` structurally load-bearing in the
                  decode again (plain decode leaves it with zero effect on
                  `top_k_tokens`). Not recommended as a decode mode - its required
                  random_control does not land at chance at any candidate_size tested -
                  kept only for reproducing that diagnosis.

Usage:
    # Build direction vectors (once, shared across genders):
    python3 elicitation_methods/gender_direction.py --mode build --layers 11

    # Decode a rollout file at that layer:
    python3 elicitation_methods/gender_direction.py --mode apply \\
        --model_name <model> --data_file <rollouts.json> \\
        --direction_file <direction_*.npy> --layer 11
"""
import argparse
import json
import os
import sys
from contextlib import ExitStack
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


# --- shared -------------------------------------------------------------------------

def load_direction(path: str) -> np.ndarray:
    vec = np.load(path).astype(np.float32)
    return vec / np.linalg.norm(vec)


# --- build mode -----------------------------------------------------------------------

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


def run_build(args):
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


# --- apply mode -----------------------------------------------------------------------

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
        #
        # Multiply in the model's own bfloat16 and cast only afterwards, exactly as
        # residual_tokens.py does. This matters: earlier revisions cast to float32 first,
        # which reorders the top-k on ~50% of items (bf16 has an 8-bit mantissa, and
        # cosine similarities over a 256k vocabulary are packed tightly enough for that to
        # matter). It barely changes their gender-word content, but it did make this
        # method's decode silently different from the one it is being compared against.
        # With the arithmetic matched, plain mode reproduces Acts Tokens byte for byte.
        residual_similarities = (
            F.normalize(residual_reps, dim=1)
            @ F.normalize(token_embeddings, dim=1).T
        ).to(dtype=torch.float32).cpu().numpy()

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


def run_apply(args):
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


# --- entrypoint -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a diff-of-means gender direction, or apply one to rollouts.")
    parser.add_argument("--mode", type=str, required=True, choices=["build", "apply"])
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                         help="build: BASE (non-fine-tuned) model to probe. apply: secret-keeping model to read "
                              "activations from (or the base model, for the base-model control). Required for apply.")
    parser.add_argument("--seed", type=int, default=None, help="build defaults to 1 if unset; apply defaults to unset.")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="build default: user_gender/inputs/gender_direction. "
                              "apply default: user_gender/results/gender_direction_tokens.")

    build_group = parser.add_argument_group("--mode build")
    build_group.add_argument("--pairs_file", type=str, default="prompts/gender/contrastive_pairs.jsonl")
    build_group.add_argument("--layers", type=int, nargs="+", default=[21, 22, 23, 24, 25],
                              help="Layers to compute a direction for (paper uses layer 23 for User Gender).")
    build_group.add_argument("--random_control", action="store_true",
                              help="Ignore contrastive data; save a random unit vector per layer instead "
                                   "(required control).")

    apply_group = parser.add_argument_group("--mode apply")
    apply_group.add_argument("--data_file", type=str, default=None)
    apply_group.add_argument("--direction_file", type=str, default=None,
                              help="Path to a direction_*.npy file from --mode build.")
    apply_group.add_argument("--layer", type=int, default=None)
    apply_group.add_argument("--top_k", type=int, default=200)
    apply_group.add_argument("--candidate_size", type=int, default=None,
                              help="Enable blend decode: restrict the decode candidate vocabulary to this many "
                                   "tokens most aligned with the per-item signed direction, then rank those by "
                                   "the real residual. Omit for plain full-vocab residual decode (direction has "
                                   "no effect on top_k_tokens in that mode). Must be >= --top_k.")
    apply_group.add_argument("--limit", type=int, default=None,
                              help="Process only the first N pairs - use for a quick pilot before a full run.")

    args = parser.parse_args()
    load_dotenv()

    if args.mode == "build":
        if args.output_dir is None:
            args.output_dir = "user_gender/inputs/gender_direction"
        if args.seed is None:
            args.seed = 1
        run_build(args)
    else:
        for required in ("data_file", "direction_file", "layer"):
            if getattr(args, required) is None:
                print(f"Error: --{required} is required for --mode apply")
                sys.exit(1)
        if args.output_dir is None:
            args.output_dir = "user_gender/results/gender_direction_tokens"
        run_apply(args)


if __name__ == "__main__":
    main()
