#!/usr/bin/env python3
"""Per-item projection on the gender direction by true target gender, one layer and prompt set, per variant.

Draws derived/user_gender/projection_values_layer{layer}.json (written by analysis/projection_values.py) as styles-plot
strips (each item a dot, the median a line; violins are not in the grammar): one block per variant, rows = the two
checkpoints, one shared x axis; the contrastive direction in its role colour, the control variants grey.
Figure: figures/post/projection_separability_layer{layer}_{prompt_set}[_contrastive_only].png.
"""
import argparse

DEFAULT_VARIANTS = ["contrastive", "base_model_control"]
GENDERS = ["female", "male"]
VARIANT_TERMS = {"contrastive": "fine-tuned checkpoints", "random_control": "random control", "base_model_control": "base model"}
VARIANT_CONTROLS = {"random_control": "a random direction", "base_model_control": "the base model"}


def figure_id(layer: int, prompt_set: str, variants) -> str:
    """The ids keep the old PNG names: *_contrastive_only for the contrastive variant alone."""
    suffix = "_contrastive_only" if list(variants) == ["contrastive"] else ""
    return f"projection_separability_layer{layer}_{prompt_set}{suffix}"


def draw(raw, layer, prompt_set, out_dir):
    """One strip block per variant (each prompt a dot, a median line), rows = the two checkpoints, one shared
    x axis; the contrastive direction in its role colour, the control variants grey."""
    import styles_plot as sp
    from figstyle import save, vocab

    allv = [x for by_g in raw.values() for vals in by_g.values() for x in vals]
    lo, hi = min(allv), max(allv)
    pad = 0.05 * (hi - lo)
    xlim = (lo - pad, hi + pad)
    fig = sp.Figure(figure_id(layer, prompt_set, raw), vocab())
    for variant, by_gender in raw.items():
        col = VARIANT_TERMS[variant]
        role = "method" if variant == "contrastive" else "control"
        fig.add(sp.Distribution(GENDERS, "projection", {(g, col): by_gender[g] for g in GENDERS}, role=role,
                                control=VARIANT_CONTROLS.get(variant), kind="checkpoints", style="strip", median=True,
                                columns=[col], xlim=xlim, panel_h=0.55,
                                cells={(g, col): f"dist.{variant}.{g}" for g in GENDERS}))
    return save(fig, out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    values = derived(f"projection_values_layer{args.layer}.json")[args.prompt_set]
    draw({v: values[v] for v in args.variants}, args.layer, args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
