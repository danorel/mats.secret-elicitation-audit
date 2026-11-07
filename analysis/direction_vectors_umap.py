#!/usr/bin/env python3
"""Requires: CPU. UMAP of the gender_direction vectors themselves, colored by variant.

Loads every saved direction under user_gender/inputs/gender_direction/
{contrastive,random_control}/direction_*_layer*.npy (layers 10-25, both variants, 32
vectors of 3584 dims each), projects to 2D. No GPU - vectors are already on disk.

Question: do real contrastive directions form a coherent trajectory across layers and
separate from random-vector controls, or do both variants look alike here?

Writes JSON only; direction_vectors_umap_chart.py draws it (no umap/sklearn needed there).
"""
import argparse
import os

import numpy as np

import runs  # noqa: E402
from meta import write_derived  # noqa: E402


def load_vectors():
    vectors, meta = [], []
    e = runs.entry("directions.l10to25")
    for variant in e["variants"]:
        for layer in e["layers"]:
            path = runs.open_path("directions.l10to25", variant=variant, layer=layer)
            vec = np.load(path)
            vectors.append(vec / np.linalg.norm(vec))  # unit-normalize: only direction matters
            meta.append({"variant": variant, "layer": layer, "file": os.path.basename(path)})
    return np.stack(vectors), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_neighbors", type=int, default=8,
                         help="Lowered from UMAP's default (15) - only 32 points total.")
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    vectors, meta = load_vectors()
    print(f"Loaded {len(vectors)} unit-normalized direction vectors ({vectors.shape[1]}-dim)")

    import umap
    from sklearn.decomposition import PCA

    reducer = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                         random_state=args.seed, metric="cosine")
    coords_umap = reducer.fit_transform(vectors)

    pca = PCA(n_components=2, random_state=args.seed)
    coords_pca = pca.fit_transform(vectors)

    points = []
    for i, m in enumerate(meta):
        points.append({
            **m,
            "umap_x": float(coords_umap[i, 0]), "umap_y": float(coords_umap[i, 1]),
            "pca_x": float(coords_pca[i, 0]), "pca_y": float(coords_pca[i, 1]),
        })

    write_derived("direction_vectors_umap.json",
                  {"points": points, "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, __file__)
    print("Figure: python3 plots/direction_vectors_umap_chart.py")


if __name__ == "__main__":
    main()
