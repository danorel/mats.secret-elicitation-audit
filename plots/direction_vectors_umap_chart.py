#!/usr/bin/env python3
"""Figure: UMAP and PCA of the gender-direction vectors, contrastive vs. random control (no GPU, no umap).

Reads direction_vectors_umap.py's JSON (direction_vectors_umap.json) and draws two styles-plot embedding panels.
The contrastive directions are joined in layer order by a grey connector with words at its two ends (the first and
last layer swept); per-point layer numbers are not drawn (sixteen labels collide at the post target's text size).
The PCA panel's header prints the two components' shares of variance.

Usage:
    python3 plots/direction_vectors_umap_chart.py
"""
import argparse



def draw(points, pca_shares, out_dir):
    import styles_plot as sp
    from figstyle import save, vocab

    v = vocab()
    contrastive = sorted((p for p in points if p["variant"] == "contrastive"), key=lambda p: p["layer"])
    random = [p for p in points if p["variant"] == "random_control"]
    ends = (f"layer {contrastive[0]['layer']}", f"layer {contrastive[-1]['layer']}")
    panels = []
    for name, key in (("UMAP", "umap"), ("PCA", "pca")):
        xy = lambda pts: [[p[f"{key}_x"], p[f"{key}_y"]] for p in pts]  # noqa: E731
        header, numbers = "", {}
        if key == "pca":
            header = "PCA (PC1 {a}, PC2 {b})"
            numbers = {"a": (pca_shares[0], "variance_share", "umap.pca.pc1_share"),
                       "b": (pca_shares[1], "variance_share", "umap.pca.pc2_share")}
        panels.append(sp.ScatterPanel([
            sp.Cloud("contrastive", "method", xy(contrastive), cell=f"{key}.contrastive", path=True, ends=ends),
            sp.Cloud("a random direction", "control", xy(random), cell=f"{key}.random_control"),
        ], embedding=name, header=header, numbers=numbers))
    return save(sp.Figure("direction_vectors_umap", v).add(sp.Scatter(panels, panel_h=3.2)), out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    from figstyle import derived

    data = derived("direction_vectors_umap.json")
    draw(data["points"], data["pca_explained_variance_ratio"], args.out_dir)


if __name__ == "__main__":
    main()
