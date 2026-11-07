#!/usr/bin/env python3
"""Patchscopes' attribute-extraction recipe (Ghandeharioun et al. 2024), applied to the
User Gender organism.

Patches a REAL hidden state - the mean residual at the first-person-pronoun position(s) in
an actual rollout, `r(item)`, no synthetic vector of any kind - into a natural-language
attribute-extraction target prompt ("X's gender is", zero-shot), and lets the target model
generate a free-text continuation. Source layer and target layer are swept independently
(source = where the residual is captured; target = where it's patched in), since the two
need not match for the target model to have enough remaining computation to verbalize the
patched content.

Two checks are required before trusting any result from this script:

  1. Compare against the no-patch baseline (printed automatically) - the raw target template
     run through the same generation model with no hook installed. If per-item results look
     similar to that baseline regardless of which item's residual was patched, the template
     is leaking its own default completion, not decoding the patch.
  2. Be deliberate about --generate_model_name. Generating from the fine-tuned secret-keeping
     checkpoint risks its own trained refusal behavior dominating any template that names the
     secret topic, independent of the patch. Generating from the base model avoids that
     specific failure mode but is not automatically clean either - a target template naming a
     sensitive attribute can still trigger the base model's own instruction-tuned hedging
     about that attribute, again independent of the patch. Check (1) either way.

Usage:
    python3 elicitation_methods/patchscope.py \\
        --model_name bcywinski/gemma-2-9b-it-user-female \\
        --data_file user_gender/results/female/direct/inference/*.json \\
        --source_layers 9 11 14 --target_layers 11 16 20 24 --limit 20
"""
import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product

import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.residual_tokens import ResidualCaptureHook
from sampling.utils import load_model_and_tokenizer
from utils.utils import find_first_person_pronoun_positions, find_second_start_of_turn_position, load_results

TARGET_TEMPLATE = "X's gender is"
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


def extract_source_residual(model, tokenizer, user_prompt, model_response, source_layer):
    """Real per-item residual at the pronoun position(s) - identical extraction to every
    other method in this project, no synthetic construction."""
    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": model_response}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    tokens = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").squeeze(0)
    start = find_second_start_of_turn_position(tokens, tokenizer)
    positions = find_first_person_pronoun_positions(tokens, tokenizer, start)
    if not positions:
        return None
    with ResidualCaptureHook(model, source_layer) as hook:
        with torch.no_grad():
            _ = model(tokens.unsqueeze(0).to(model.device))
        reps = hook.representations[0].to(torch.float32)
    return reps[positions, :].mean(dim=0)


def patch_and_generate(model, tokenizer, vec, target_layer, interp_ids, patch_pos, max_new_tokens):
    v = vec.to(model.device).unsqueeze(0)
    with PatchDirectionHook(model, target_layer, patch_pos, v):
        with torch.no_grad():
            gen_out = model.generate(
                interp_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
    return tokenizer.decode(gen_out[0][interp_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, required=True,
                         help="Checkpoint to extract the REAL source residual from.")
    parser.add_argument("--generate_model_name", type=str, default=None,
                         help="Checkpoint to patch into and generate from. Defaults to --model_name. "
                              "See module docstring for why the fine-tuned checkpoint and even the base "
                              "model both risk contaminating this specific template.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--source_layers", type=int, nargs="+", default=[11])
    parser.add_argument("--target_layers", type=int, nargs="+", default=[11, 16, 18, 20, 24])
    parser.add_argument("--target_template", type=str, default=TARGET_TEMPLATE)
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--placeholder", type=str, default=PLACEHOLDER,
                         help="Subject placeholder to locate and patch within --target_template. "
                              "Must tokenize to a single token (checked at runtime).")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    print(f"Loading source model: {args.model_name}")
    source_model, source_tokenizer = load_model_and_tokenizer(args.model_name)

    gen_name = args.generate_model_name or args.model_name
    if gen_name == args.model_name:
        gen_model, gen_tokenizer = source_model, source_tokenizer
    else:
        print(f"Loading generation model: {gen_name}")
        gen_model, gen_tokenizer = load_model_and_tokenizer(gen_name)

    interp_ids = torch.tensor(
        gen_tokenizer(args.target_template, add_special_tokens=True)["input_ids"]
    ).unsqueeze(0).to(gen_model.device)
    patch_pos = find_placeholder_position(gen_tokenizer, args.target_template, args.placeholder)
    print(f"Target template: {args.target_template!r} (patch_pos={patch_pos}, len={interp_ids.shape[1]})")

    no_patch_ids = interp_ids
    with torch.no_grad():
        no_patch_out = gen_model.generate(no_patch_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           pad_token_id=gen_tokenizer.eos_token_id)
    no_patch_decoded = gen_tokenizer.decode(no_patch_out[0][no_patch_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"REQUIRED SANITY CHECK - no-patch completion of the raw template on {gen_name}: "
          f"{no_patch_decoded!r}\n(If per-item results below look similar to this regardless of source item, "
          f"they are not real decoding - see docstring.)")

    pairs = load_results(args.data_file)
    if args.limit:
        pairs = pairs[:args.limit]

    rows = []
    for source_layer in args.source_layers:
        sources = []
        for pair in tqdm(pairs, desc=f"extracting @ source layer {source_layer}"):
            vec = extract_source_residual(source_model, source_tokenizer, pair.get("user_prompt", ""),
                                           pair.get("model_response", ""), source_layer)
            if vec is not None:
                sources.append((pair.get("user_prompt", ""), vec))

        for target_layer, (prompt, vec) in tqdm(list(product(args.target_layers, sources)),
                                                  desc=f"patching source L{source_layer} -> target layers"):
            decoded = patch_and_generate(gen_model, gen_tokenizer, vec, target_layer, interp_ids, patch_pos,
                                          args.max_new_tokens)
            rows.append({"user_prompt": prompt, "source_layer": source_layer,
                         "target_layer": target_layer, "decoded": decoded})

    report = {
        "model_name": args.model_name, "generate_model_name": gen_name,
        "no_patch_baseline_completion": no_patch_decoded, "data_file": args.data_file,
        "target_template": args.target_template, "source_layers": args.source_layers,
        "target_layers": args.target_layers, "timestamp": datetime.now().isoformat(),
        "rows": rows,
    }
    out = args.output_json or "results/user_gender/patchscope_attribute_extraction.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
