#!/usr/bin/env python3
"""UMAP of the gender_direction vectors themselves, colored by variant.

Loads every saved direction under user_gender/inputs/gender_direction/
{contrastive,random_control}/direction_*_layer*.npy (layers 10-25, both variants, 32
vectors of 3584 dims each), projects to 2D. No GPU - vectors are already on disk.

Question: do real contrastive directions form a coherent trajectory across layers and
separate from random-vector controls, or do both variants look alike here?
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load_vectors(inputs_dir: str):
    vectors, meta = [], []
    for variant in ["contrastive", "random_control"]:
        pattern = os.path.join(inputs_dir, variant, f"direction_{variant}_layer*.npy")
        for path in sorted(glob.glob(pattern)):
            layer = int(re.search(r"layer(\d+)", os.path.basename(path)).group(1))
            vec = np.load(path)
            vectors.append(vec / np.linalg.norm(vec))  # unit-normalize: only direction matters
            meta.append({"variant": variant, "layer": layer, "file": os.path.basename(path)})
    return np.stack(vectors), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs_dir", type=str, default="user_gender/inputs/gender_direction")
    parser.add_argument("--n_neighbors", type=int, default=8,
                         help="Lowered from UMAP's default (15) - only 32 points total.")
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_json", type=str,
                         default="user_gender/plots/direction_vectors_umap.json")
    parser.add_argument("--output_png", type=str,
                         default="user_gender/plots/direction_vectors_umap.png")
    args = parser.parse_args()

    vectors, meta = load_vectors(args.inputs_dir)
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

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"points": points, "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, f, indent=2)
    print(f"Wrote {args.output_json}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, xkey, ykey, title in [
            (axes[0], "umap_x", "umap_y", "UMAP"),
            (axes[1], "pca_x", "pca_y", f"PCA (PC1 {pca.explained_variance_ratio_[0]:.0%}, PC2 {pca.explained_variance_ratio_[1]:.0%})"),
        ]:
            for variant, color, marker in [("contrastive", "#c2185b", "o"), ("random_control", "#9e9e9e", "x")]:
                pts = [p for p in points if p["variant"] == variant]
                ax.scatter([p[xkey] for p in pts], [p[ykey] for p in pts],
                           c=color, marker=marker, s=60, label=variant, alpha=0.85)
                if variant == "contrastive":
                    order = sorted(pts, key=lambda p: p["layer"])
                    ax.plot([p[xkey] for p in order], [p[ykey] for p in order],
                            color=color, alpha=0.3, linewidth=1, zorder=0)
                    for p in pts:
                        ax.annotate(str(p["layer"]), (p[xkey], p[ykey]), fontsize=7,
                                    xytext=(3, 3), textcoords="offset points")
            ax.set_title(title)
            ax.legend()
        fig.suptitle("Gender Direction vectors (unit-normalized), layers 10-25: contrastive vs. random_control")
        fig.tight_layout()
        fig.savefig(args.output_png, dpi=150)
        print(f"Wrote {args.output_png}")
    except ImportError:
        print("matplotlib not available, skipping PNG")


if __name__ == "__main__":
    main()
