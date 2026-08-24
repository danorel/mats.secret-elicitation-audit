#!/usr/bin/env python3
"""Static matplotlib scatter of umap_contrastive_pairs.py's output (UMAP + PCA side by side).

No GPU needed - just reads the JSON that umap_contrastive_pairs.py already saved.
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Plot UMAP/PCA coordinates from umap_contrastive_pairs.py.")
    parser.add_argument("--input_json", type=str,
                         default="user_gender/inputs/gender_direction/umap/contrastive_umap_layer24_google_gemma-2-9b-it.json")
    parser.add_argument("--output_png", type=str, default="plots/output/contrastive_umap.png")
    parser.add_argument("--label_points", action="store_true",
                         help="Annotate each point with its sentence (readable only for small n).")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = data["points"]
    meta = data["metadata"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"female": "#2a78d6", "male": "#eb6834"}

    fig, (ax_umap, ax_pca) = plt.subplots(1, 2, figsize=(13, 6))

    for ax, xkey, ykey, title in [
        (ax_umap, "umap_x", "umap_y", f"UMAP (n_neighbors={meta['umap_n_neighbors']}, min_dist={meta['umap_min_dist']}, cosine)"),
        (ax_pca, "pca_x", "pca_y", "PCA (linear, sanity check)"),
    ]:
        for label in ["female", "male"]:
            xs = [p[xkey] for p in points if p["label"] == label]
            ys = [p[ykey] for p in points if p["label"] == label]
            ax.scatter(xs, ys, c=colors[label], label=label, s=45, alpha=0.85,
                       edgecolors="white", linewidths=0.6)
            if args.label_points:
                for p in points:
                    if p["label"] != label:
                        continue
                    ax.annotate(p["sentence"][:24], (p[xkey], p[ykey]), fontsize=6,
                                alpha=0.6, xytext=(3, 3), textcoords="offset points")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Contrastive-pair activations · layer {meta['layer']} · {meta['model_name']} · "
        f"{meta['n_pairs_used']} pairs ({meta['hidden_dim']}-d → 2D)",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    fig.savefig(args.output_png, dpi=160)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
