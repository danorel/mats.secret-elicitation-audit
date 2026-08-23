#!/usr/bin/env python3
"""UMAP a 2D projection of the raw contrastive-pair activations used to build gender_direction.

build_gender_direction.py only ever kept the *mean* of the female cloud minus the mean of
the male cloud - the individual per-sentence activation vectors were never saved. This script
re-runs the identical extraction (same model, same pronoun-position logic, imported directly
from build_gender_direction.py so the two stay in lockstep) but keeps every per-pair vector,
then fits UMAP to project them to 2D for visualization.

Output is two files: a .npz with the raw high-dimensional vectors (so UMAP can be re-fit with
different params without reloading the model) and a small .json with the 2D coordinates +
sentence text, sized for a browser visualization.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elicitation_methods.build_gender_direction import get_pronoun_activations, load_contrastive_pairs
from sampling.utils import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="UMAP the raw contrastive-pair activations for gender_direction.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it",
                         help="BASE (non-fine-tuned) model to probe - must match build_gender_direction.py's default "
                              "so the points correspond to the same direction vectors already built.")
    parser.add_argument("--pairs_file", type=str, default="prompts/gender/contrastive_pairs.jsonl")
    parser.add_argument("--layer", type=int, default=24, help="Single layer to embed (best layer from the sweep).")
    parser.add_argument("--n_neighbors", type=int, default=10,
                         help="UMAP n_neighbors. Default lowered from the library default (15) because there are "
                              "only ~50 pairs (~100 points) - too few for a neighborhood of 15.")
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="user_gender/results_contrastive/gender_direction/umap")
    args = parser.parse_args()

    load_dotenv()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading BASE model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    pairs = load_contrastive_pairs(args.pairs_file)
    print(f"Loaded {len(pairs)} contrastive pairs from {args.pairs_file}")

    male_vecs, female_vecs, male_sents, female_sents = [], [], [], []
    skipped = 0
    for pair in tqdm(pairs, desc="Extracting activations"):
        m_acts = get_pronoun_activations(model, tokenizer, pair["male"], [args.layer])
        f_acts = get_pronoun_activations(model, tokenizer, pair["female"], [args.layer])
        if not m_acts or not f_acts:
            skipped += 1
            continue
        male_vecs.append(m_acts[args.layer])
        female_vecs.append(f_acts[args.layer])
        male_sents.append(pair["male"])
        female_sents.append(pair["female"])

    if skipped:
        print(f"Skipped {skipped}/{len(pairs)} pairs (no first-person pronoun found on one side).")

    male_vecs = np.stack(male_vecs, axis=0)
    female_vecs = np.stack(female_vecs, axis=0)
    all_vecs = np.concatenate([female_vecs, male_vecs], axis=0)
    labels = ["female"] * len(female_vecs) + ["male"] * len(male_vecs)
    sentences = female_sents + male_sents

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = args.model_name.replace("/", "_")

    npz_path = os.path.join(args.output_dir, f"contrastive_activations_layer{args.layer}_{model_tag}.npz")
    np.savez(npz_path, female_vecs=female_vecs, male_vecs=male_vecs,
             female_sentences=np.array(female_sents), male_sentences=np.array(male_sents))
    print(f"Saved raw activations to {npz_path}")

    print("Fitting UMAP...")
    import umap
    n_neighbors = min(args.n_neighbors, len(all_vecs) - 1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=args.min_dist,
                         random_state=args.seed, metric="cosine")
    coords = reducer.fit_transform(all_vecs)

    # Also fit a 2D PCA on the same vectors as a cheap, non-stochastic sanity check that UMAP's
    # layout isn't purely an artifact of its own stochastic optimization on so few points.
    from sklearn.decomposition import PCA
    pca_coords = PCA(n_components=2, random_state=args.seed).fit_transform(all_vecs)

    points = []
    for i in range(len(all_vecs)):
        points.append({
            "label": labels[i],
            "sentence": sentences[i],
            "umap_x": float(coords[i, 0]),
            "umap_y": float(coords[i, 1]),
            "pca_x": float(pca_coords[i, 0]),
            "pca_y": float(pca_coords[i, 1]),
        })

    out = {
        "metadata": {
            "model_name": args.model_name,
            "pairs_file": args.pairs_file,
            "layer": args.layer,
            "n_pairs_used": len(female_vecs),
            "n_pairs_skipped": skipped,
            "hidden_dim": int(all_vecs.shape[1]),
            "umap_n_neighbors": n_neighbors,
            "umap_min_dist": args.min_dist,
            "umap_metric": "cosine",
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "points": points,
    }

    json_path = os.path.join(args.output_dir, f"contrastive_umap_layer{args.layer}_{model_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved UMAP + PCA coordinates to {json_path}")


if __name__ == "__main__":
    main()
