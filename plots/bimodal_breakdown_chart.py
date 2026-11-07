#!/usr/bin/env python3
"""Per-prompt auditor accuracy of the layer-11 readout, one prompt set: a histogram per checkpoint.

Draws derived/user_gender/bimodal_breakdown_{prompt_set}.json (written by analysis/bimodal_breakdown.py).
Figure: figures/post/bimodal_breakdown_{prompt_set}.png.
"""
import argparse

GENDERS = ["female", "male"]
BINS = [i / 10 for i in range(11)]


def draw(per_gender, prompt_set, out_dir):
    """Rows = the two checkpoints: a histogram of per-prompt accuracy in tenths."""
    import styles_plot as sp
    from figstyle import save, vocab

    d = sp.Distribution(GENDERS, "prompt_accuracy", per_gender, role="method", kind="checkpoints",
                        cells={g: f"dist.{g}" for g in GENDERS}, count_unit="prompts", bins=BINS, xlim=(0, 1),
                        xticks=[0, 0.5, 1])
    return save(sp.Figure(f"bimodal_breakdown_{prompt_set}", vocab()).add(d), out_dir)


def main():
    from figstyle import derived

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt_set", type=str, default="direct", choices=["direct", "standard"])
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/ at the repo root")
    args = parser.parse_args()
    draw(derived(f"bimodal_breakdown_{args.prompt_set}.json"), args.prompt_set, args.out_dir)


if __name__ == "__main__":
    main()
