#!/usr/bin/env python3
"""Patchscope-style natural-language decoding of the gender direction (Hypothesis D2).

Session 2 found that every method decoding into raw token/embedding space (LL Tokens,
Acts Tokens, SAE Tokens, gender_direction/apply_gender_direction.py) tops out around
60-70% auditor accuracy on `direct`, while SAE Desc - the one method that decodes into
natural-language feature descriptions instead - reaches 87.8%. This script tests whether
that gap is about the *decode format* (raw tokens vs. natural language) rather than
something specific to SAE: it takes the same signed gender_direction projection used by
apply_gender_direction.py, but instead of nearest-token lookup, patches it into a fresh
forward pass (Patchscopes, Ghandeharioun et al. 2024) with an induction few-shot prompt
and lets the model itself generate a free-text description, which is fed to the auditor
the same way SAE Desc's feature descriptions are.

Template history: this originally used a *definitional* prompt ("dog: a loyal
four-legged..."), which session 3 found breaks even for real word embeddings by
layer 10-15, and session 4's full 0-32 layer sweep confirmed breaks again by layer 24.
Switched to the induction/copying template below ("dog -> dog\n..."), which session 4
validated works cleanly on real words across layers 16-20 - and, going further,
directly on both `bcywinski/gemma-2-9b-it-user-{female,male}` (not just the base
model) - yet still never surfaces gender-related content for `gender_direction` itself
anywhere in that range. See log/2026-08-24-session-04.md for the full audit trail;
earlier saved layer-24 output under this script used the old definitional prompt and
should not be cited as induction-template evidence.

Reuses apply_gender_direction.py's `load_direction` and the same pronoun-position /
signed-projection extraction logic (duplicated here rather than imported, to keep that
file untouched - see CLAUDE.md "keep new code additive").
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.apply_gender_direction import load_direction
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import (
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    load_results,
)

# Induction/copying continuation template (the classic Patchscopes identity-decode
# prompt) - validated in session 4 to work cleanly on real word embeddings across
# layers 16-20, on both the base model and both fine-tuned checkpoints (see module
# docstring). "X" is the position whose hidden state gets overwritten with the
# (scaled) direction at generation time; the model's own continuation after "X ->"
# is the natural-language decode.
INTERPRETATION_PROMPT = "dog -> dog\nocean -> ocean\nhappiness -> happiness\nX ->"
PLACEHOLDER = "X"


def find_placeholder_position(tokenizer, full_text: str, placeholder: str) -> int:
    """Locate the token index of the last `placeholder` occurrence in `full_text`."""
    prefix = full_text[: full_text.rindex(placeholder)]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    pos = len(prefix_ids)
    if pos >= len(full_ids):
        raise ValueError("Placeholder position not found (tokenizer boundary mismatch).")
    decoded = tokenizer.decode([full_ids[pos]]).strip()
    if placeholder not in decoded and decoded not in placeholder:
        raise ValueError(f"Tokenizer boundary mismatch: expected '{placeholder}', got '{decoded}'.")
    return pos


class PatchDirectionHook:
    """Overwrites one token position's hidden state during the prompt prefill pass only.

    Guarded by seq_len > patch_pos so it's a no-op during incremental decoding steps (where
    the KV cache means each step only processes a length-1 tensor for the new token) - the
    single prefill-time patch is enough for its effect to propagate through the cache into
    every subsequently generated token, matching how Patchscopes intervenes.
    """

    def __init__(self, model, layer_idx: int, patch_pos: int, vector: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.patch_pos = patch_pos
        self.vector = vector
        self.hook_handle = None

    def _hook(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if hidden_states.shape[1] > self.patch_pos:
            hidden_states[:, self.patch_pos, :] = self.vector.to(hidden_states.dtype)
        return (hidden_states,) + output[1:] if isinstance(output, tuple) else hidden_states

    def __enter__(self):
        target_layer_module = self.model.model.layers[self.layer_idx]
        self.hook_handle = target_layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()


def extract_signed_vector(model, tokenizer, user_prompt: str, model_response: str, target_layer: int,
                           direction_hat: torch.Tensor) -> Dict[str, Any]:
    """Per-item forward pass only (cheap - no generation). Returns a dict that either has
    `signed_vector` (a float32 hidden_dim tensor, ready to patch) or an `error` string."""
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
        }

    try:
        with ResidualCaptureHook(model, target_layer) as hook:
            with torch.no_grad():
                _ = model(tokens.unsqueeze(0).to(model.device))
            residual_reps = hook.representations[0]  # (seq_len, hidden_dim)

        direction_hat_dev = direction_hat.to(device=residual_reps.device, dtype=torch.float32)

        pronoun_projections = [
            torch.dot(residual_reps[pos].to(torch.float32), direction_hat_dev).item() for pos in pronoun_positions
        ]
        pronoun_norms = [torch.norm(residual_reps[pos].to(torch.float32)).item() for pos in pronoun_positions]

        mean_proj = float(np.mean(pronoun_projections))
        injection_scale = float(np.mean(pronoun_norms))
        signed_direction = direction_hat_dev if mean_proj >= 0 else -direction_hat_dev

        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "start_of_turn_position": int(start_of_turn_position) if start_of_turn_position != -1 else None,
            "pronoun_positions": [int(p) for p in pronoun_positions],
            "mean_projection": mean_proj,
            "injection_scale": injection_scale,
            "signed_vector": (signed_direction * injection_scale).cpu(),
        }
    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "pronoun_positions": [int(p) for p in pronoun_positions],
        }


def batch_decode(model, tokenizer, items: List[Dict[str, Any]], target_layer: int, interp_ids: torch.Tensor,
                  patch_pos: int, max_new_tokens: int, batch_size: int) -> None:
    """Patches each item's `signed_vector` into the (fixed) interpretation prompt and
    generates, `batch_size` items at a time in one `generate()` call - this is the step
    that was previously one `generate()` call per item (the actual cause of session 3's
    3-hour run: unbatched autoregressive decoding, not GPU or method slowness)."""
    valid = [it for it in items if "signed_vector" in it]
    for start in tqdm(range(0, len(valid), batch_size), desc="Patchscope-decoding (batched)"):
        batch = valid[start:start + batch_size]
        b = len(batch)
        vectors = torch.stack([it["signed_vector"] for it in batch]).to(model.device)
        interp_batch = interp_ids.repeat(b, 1).to(model.device)
        with PatchDirectionHook(model, target_layer, patch_pos, vectors):
            with torch.no_grad():
                gen_out = model.generate(
                    interp_batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        for i, it in enumerate(batch):
            it["patchscope_description"] = tokenizer.decode(
                gen_out[i][interp_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        for it in batch:
            del it["signed_vector"]


def main():
    parser = argparse.ArgumentParser(description="Patchscope natural-language decode of gender_direction.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--direction_file", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=25,
                         help="Items per generate() call in the decode phase. Higher = faster but more VRAM.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N pairs - use this for a quick pilot (10-20) before a "
                              "full run. A full run takes ~15-20 min per file with batched decoding; a pilot "
                              "with --limit 20 takes well under a minute.")
    parser.add_argument("--output_dir", type=str, default="user_gender/results/gender_direction_patchscope")
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

    interp_ids = torch.tensor(
        tokenizer(INTERPRETATION_PROMPT, add_special_tokens=True)["input_ids"]
    ).unsqueeze(0)
    patch_pos = find_placeholder_position(tokenizer, INTERPRETATION_PROMPT, PLACEHOLDER)

    input_output_pairs = load_results(args.data_file)
    if not input_output_pairs:
        print("No input-output pairs found. Exiting.")
        sys.exit(1)
    if args.limit is not None:
        input_output_pairs = input_output_pairs[:args.limit]
        print(f"--limit {args.limit}: pilot run on {len(input_output_pairs)} pairs")

    # Phase 1: per-item forward pass to get each item's signed vector (cheap, no generation).
    all_results = []
    for pair_data in tqdm(input_output_pairs, desc="Extracting signed vectors"):
        user_prompt = pair_data.get("user_prompt", "")
        model_response = pair_data.get("model_response", "")
        response_idx = pair_data.get("model_response_index", 0)

        result = extract_signed_vector(model, tokenizer, user_prompt, model_response, args.layer, direction_hat)
        result["response_index"] = response_idx
        result["layer"] = args.layer
        all_results.append(result)

    # Phase 2: batched patch + generate for every item that got a valid signed vector.
    batch_decode(model, tokenizer, all_results, args.layer, interp_ids, patch_pos,
                 args.max_new_tokens, args.batch_size)

    processed, errors = 0, 0
    for result in all_results:
        if "error" in result:
            errors += 1
            result["patchscope_description"] = ""
        else:
            processed += 1

    results = {
        "metadata": {
            "model_name": args.model_name,
            "source_data_file": args.data_file,
            "source_filename": os.path.basename(args.data_file),
            "direction_file": args.direction_file,
            "target_layer": args.layer,
            "max_new_tokens": args.max_new_tokens,
            "interpretation_prompt": INTERPRETATION_PROMPT,
            "mode": "gender_direction_patchscope",
            "method": "patchscope_direction_decode",
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
    filename = (f"gender_direction_patchscope_layer{args.layer}_{direction_tag}_"
                f"{model_name_safe}_on_{inference_filename}{seed_suffix}_{timestamp}.json")
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Processed {processed}/{len(input_output_pairs)} pairs successfully ({errors} errors)")


if __name__ == "__main__":
    main()
