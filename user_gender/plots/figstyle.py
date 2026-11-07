"""styles-plot setup shared by the chart scripts in user_gender/plots/.

One vocabulary (plotstyle.toml at the repo root), one status registry (figures/status.csv), one output folder
(figures/<target>/ at the repo root). The numbers the figures read live under results/user_gender/. Saves are
strict: a figure that breaks the grammar raises styles_plot.StyleError and nothing is written.
"""
import os

import styles_plot as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOCAB_PATH = os.path.join(ROOT, "plotstyle.toml")
FIG_DIR = os.path.join(ROOT, "figures")
TARGET = "post"


def vocab() -> sp.Vocab:
    return sp.load(VOCAB_PATH)


def save(fig: sp.Figure, out_dir: str | None = None, target: str = TARGET) -> str:
    """Write <out_dir>/<target>/<figure id>.png (out_dir defaults to figures/ at the repo root) and its
    .styles.json sidecar."""
    path = os.path.join(out_dir or FIG_DIR, target, f"{fig.id}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sp.save(fig, path, target=target)
    print(f"Wrote {os.path.relpath(path, ROOT)}")
    return path
