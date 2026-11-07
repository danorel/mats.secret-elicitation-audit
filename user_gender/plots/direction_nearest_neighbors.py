#!/usr/bin/env python3
"""Does a direction vector have clean nearest neighbors in embedding space, independent of
any generative decode mechanism?

Raw cosine similarity between the direction and every row of the token embedding matrix - no
forward pass beyond loading the model, no generation, the same operation
`gender_direction.py`'s own per-item decode uses (`F.normalize(...) @
F.normalize(token_embeddings, ...).T`), applied to the bare direction instead of a per-item
residual.

Includes the existing Gaussian `random_control` direction as a required control: cosine
magnitudes in a 3584-dim space with a 256k-token vocabulary are small regardless of the
vector (typically 0.05-0.13), so raw magnitude proves nothing - the discriminator is whether
the top-k list is *thematically coherent* or scattered, checked against a vector known to
carry no structure.

No GPU compute beyond loading the model once (a single embedding matrix lookup per vector).

Usage:
    python3 user_gender/plots/direction_nearest_neighbors.py --layer 11
    python3 user_gender/plots/direction_nearest_neighbors.py --vector_file path/to/some_direction.npy --label "my vector"
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from elicitation_methods.gender_direction import load_direction
from sampling.utils import load_model_and_tokenizer


def nearest(direction_hat: np.ndarray, emb: torch.Tensor, tokenizer, k: int):
    v = torch.from_numpy(direction_hat).to(emb.device, torch.float32)
    sims = F.normalize(v.unsqueeze(0), dim=1) @ F.normalize(emb, dim=1).T
    top = torch.topk(sims.squeeze(0), k)
    return [{"token": tokenizer.decode([i.item()]), "cosine": round(s.item(), 4)}
            for s, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", type=str, default="bcywinski/gemma-2-9b-it-user-female",
                         help="Any Gemma-2-9b checkpoint - the embedding matrix is what matters, and it's "
                              "shared across base/fine-tuned checkpoints in this organism.")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--gender_direction_file", type=str,
                         default="user_gender/inputs/gender_direction/contrastive/direction_contrastive_layer{layer}.npy")
    parser.add_argument("--random_control_file", type=str,
                         default="user_gender/inputs/gender_direction/random_control/direction_random_control_layer{layer}.npy")
    parser.add_argument("--control_direction_files", type=str, nargs="*", default=[],
                         help="Extra (label=path) pairs, e.g. sentiment=user_gender/inputs/.../direction_contrastive_layer11.npy")
    parser.add_argument("--mean_delta_dir", type=str, default="results/user_gender/mean_delta_vectors")
    parser.add_argument("--prompt_set", type=str, default="direct")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    load_dotenv()
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    emb = model.model.embed_tokens.weight.detach().to(torch.float32)

    vectors = {}
    gender_hat = load_direction(args.gender_direction_file.format(layer=args.layer))
    vectors["gender_direction"] = gender_hat
    random_hat = load_direction(args.random_control_file.format(layer=args.layer))
    vectors["random_control_gaussian"] = random_hat

    for spec in args.control_direction_files:
        label, path = spec.split("=", 1)
        vectors[label] = load_direction(path)

    fm_path = os.path.join(args.mean_delta_dir, f"mean_delta_female_layer{args.layer}_{args.prompt_set}.npy")
    mm_path = os.path.join(args.mean_delta_dir, f"mean_delta_male_layer{args.layer}_{args.prompt_set}.npy")
    if os.path.exists(fm_path) and os.path.exists(mm_path):
        fm, mm = np.load(fm_path), np.load(mm_path)
        delta_diff = fm - mm
        vectors["actual_finetuning_delta (female-male)"] = delta_diff / np.linalg.norm(delta_diff)

    report = {"layer": args.layer, "model_name": args.model_name, "k": args.k,
              "timestamp": datetime.now().isoformat(), "vectors": {}}

    for name, vec in vectors.items():
        pos = nearest(vec, emb, tokenizer, args.k)
        neg = nearest(-vec, emb, tokenizer, args.k)
        report["vectors"][name] = {"positive": pos, "negative": neg}
        print(f"\n=== +{name} ===")
        for row in pos:
            print(f"  {row['cosine']:+.4f}  {row['token']!r}")
        print(f"=== -{name} ===")
        for row in neg:
            print(f"  {row['cosine']:+.4f}  {row['token']!r}")

    out = args.output_json or os.path.join("results", "user_gender", f"direction_nearest_neighbors_layer{args.layer}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
