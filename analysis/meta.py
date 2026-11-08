"""Derived files: where they live, how they are read, and the _meta provenance block every write adds.

    from meta import read_derived, write_derived
    rows = read_derived("reproduction_bar_chart_direct.json")       # data without _meta (lists unwrapped)
    write_derived("reproduction_bar_chart_direct.json", rows, __file__)

_meta = {script, args, git_commit, dirty, created, inputs, derived_inputs}: `inputs` are the raw runs loaded through
runs.py ({run_id, gender, kind, path, sha256}), `derived_inputs` the derived files read through read_derived
({path, sha256}). A top-level list is stored as {"_meta": ..., "rows": [...]} and read back as the list.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import runs

ROOT = runs.ROOT
DERIVED = ROOT / "derived" / "user_gender"
_DERIVED_READ: list[dict] = []


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _strip(data):
    if isinstance(data, dict) and "_meta" in data:
        data = {k: v for k, v in data.items() if k != "_meta"}
        if set(data) == {"rows"}:
            return data["rows"]
    return data


def derived_path(name: str) -> Path:
    return DERIVED / name


def read_derived(name: str):
    """A derived file's data without its _meta block (recorded in the next write's derived_inputs)."""
    p = derived_path(name)
    if not p.is_file():
        raise FileNotFoundError(f"derived/user_gender/{name} is missing; run `make numbers` (or restore it with git)")
    raw = p.read_bytes()
    rec = {"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest()}
    if rec not in _DERIVED_READ:
        _DERIVED_READ.append(rec)
    return _strip(json.loads(raw))


def write_derived(name: str, data, script: str, indent: int = 2, meta_extra: dict | None = None) -> Path:
    """Write derived/user_gender/<name> with the _meta block; `meta_extra` adds keys to _meta (provenance that is
    not a number, e.g. how an input's layout was established)."""
    dirty = bool(_git("status", "--porcelain", "--", ".", ":!derived", ":!figures/readme", ":!figures/writeup"))
    meta = {
        "script": str(Path(script).resolve().relative_to(ROOT)),
        "args": sys.argv[1:],
        "git_commit": _git("rev-parse", "HEAD"),
        "dirty": dirty,
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": runs.accessed(),
        "derived_inputs": list(_DERIVED_READ),
    }
    if meta_extra:
        clash = set(meta_extra) & set(meta)
        assert not clash, f"meta_extra would overwrite {sorted(clash)}"
        meta.update(meta_extra)
    payload = {"_meta": meta, "rows": data} if isinstance(data, list) else {"_meta": meta, **data}
    p = derived_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
    print(f"Wrote {p.relative_to(ROOT)}")
    return p
