#!/usr/bin/env python3
"""Snapshot of the build outputs, to prove a refactor changes nothing.

Records a sha256 of every PNG's decoded pixels (mode RGBA, size and pixel data, so the same image compressed by a
different zlib on another platform counts as unchanged) under figures/post/ and of every derived JSON (canonical form: sorted keys, the
`_meta` provenance block and run timestamps removed, so adding provenance does not count as a change). Derived JSON is looked for in
derived/user_gender/ and, for snapshots taken before the rename, results/user_gender/; files are keyed by name.

    python3 scripts/dev/snapshot.py save .snapshot/baseline.json
    python3 scripts/dev/snapshot.py diff .snapshot/baseline.json      # exit 1 if anything differs
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIRS = [ROOT / "derived" / "user_gender", ROOT / "results" / "user_gender"]


def canonical_json(path: Path) -> bytes:
    """The data without provenance: `_meta` and a top-level `timestamp` (a run's clock, moved into _meta.created)
    removed, and {"rows": [...]} (how a list is stored next to _meta) unwrapped to the list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if k not in ("_meta", "timestamp")}
        if set(data) == {"rows"}:
            data = data["rows"]
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def pixel_sha256(path: Path) -> str:
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        h = hashlib.sha256(f"RGBA {rgba.size[0]}x{rgba.size[1]}\n".encode())
        h.update(rgba.tobytes())
    return h.hexdigest()


def snapshot() -> dict:
    out = {"png": {}, "json": {}}
    for p in sorted((ROOT / "figures" / "post").glob("*.png")):
        out["png"][p.name] = pixel_sha256(p)
    for d in DERIVED_DIRS:
        for p in sorted(d.glob("*.json")):
            out["json"].setdefault(p.name, hashlib.sha256(canonical_json(p)).hexdigest())
    return out


def diff(old: dict, new: dict) -> list[str]:
    lines = []
    for kind in ("png", "json"):
        for name in sorted(set(old[kind]) | set(new[kind])):
            a, b = old[kind].get(name), new[kind].get(name)
            if a != b:
                lines.append(f"{kind:4s} {'added' if a is None else 'removed' if b is None else 'changed'}: {name}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["save", "diff"])
    ap.add_argument("path")
    a = ap.parse_args()
    now = snapshot()
    if a.cmd == "save":
        Path(a.path).parent.mkdir(parents=True, exist_ok=True)
        Path(a.path).write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
        print(f"saved {len(now['png'])} PNGs and {len(now['json'])} derived JSONs to {a.path}")
        return
    changes = diff(json.loads(Path(a.path).read_text()), now)
    print("\n".join(changes) if changes else f"no changes ({len(now['png'])} PNGs, {len(now['json'])} derived JSONs)")
    sys.exit(1 if changes else 0)


if __name__ == "__main__":
    main()
