"""styles-plot setup shared by the chart scripts in plots/.

One vocabulary (plotstyle.toml at the repo root), one status registry (plots/status.csv), and every figure in two
output folders at the repo root (images only; the styles-plot sidecars go to figures/_meta/<folder>/):
figures/readme/ (styles-plot's readme target, linked from README.md and EXPERIMENTS.md) and figures/writeup/ (the
post target, for the write-up). Chart scripts only draw: they read the numbers analysis/ wrote to
derived/user_gender/ (through `derived`) and nothing else. Saves are strict: a figure that breaks the grammar raises
styles_plot.StyleError and nothing is written.
"""
import json
import os

import styles_plot as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH = os.path.join(ROOT, "plotstyle.toml")
FIG_DIR = os.path.join(ROOT, "figures")
# output folder -> styles-plot target; the folder is named for where the image is used, the target sets its geometry
FOLDERS = {"readme": "readme", "writeup": "post"}
DERIVED = os.path.join(ROOT, "derived", "user_gender")


def derived(name: str):
    """A derived file's data, without its _meta provenance block (a stored {"rows": [...]} comes back as the list)."""
    path = os.path.join(DERIVED, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"derived/user_gender/{name} is missing; run `make numbers` (or restore it with git)")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "_meta" in data:
        data = {k: v for k, v in data.items() if k != "_meta"}
        if set(data) == {"rows"}:
            data = data["rows"]
    return data


def vocab() -> sp.Vocab:
    return sp.load(VOCAB_PATH)


def save(fig: sp.Figure, out_dir: str | None = None) -> list[str]:
    """Write <out_dir>/<folder>/<figure id>.png for every folder in FOLDERS (out_dir defaults to figures/ at the repo
    root), each with its .styles.json sidecar under <out_dir>/_meta/<folder>/. Returns the PNG paths."""
    root = out_dir or FIG_DIR
    paths = []
    for folder, target in FOLDERS.items():
        path = os.path.join(root, folder, f"{fig.id}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sp.save(fig, path, target=target, root=root)
        print(f"Wrote {os.path.relpath(path, ROOT)}")
        paths.append(path)
    return paths
