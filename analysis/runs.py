"""Raw runs by id, from runs.toml at the repo root. Nothing else in analysis/ builds a run path by hand.

    from runs import load
    metrics = load("acts_tokens.l11.direct", "male")              # parsed JSON of that exact file
    guesses = load("gender_direction.l11.direct", "male", kind="guessing")

Every entry names exact files (no glob), so a load either finds that one file or fails. Each load is recorded in
`accessed()` as {run_id, gender, kind, path, sha256}, which meta.py writes into the derived JSON's _meta block.
"""
import hashlib
import json
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "runs.toml"
_ACCESSED: list[dict] = []


@lru_cache(maxsize=1)
def registry() -> dict:
    with REGISTRY.open("rb") as f:
        return tomllib.load(f)["runs"]


def entry(run_id: str) -> dict:
    try:
        return registry()[run_id]
    except KeyError:
        raise KeyError(f"no run {run_id!r} in {REGISTRY.name}") from None


def path(run_id: str, gender: str | None = None, kind: str = "metrics", **fields) -> Path:
    """The exact file of a run: `kind` table keyed by gender, or a `template` filled with `fields`."""
    e = entry(run_id)
    if "template" in e:
        rel = e["template"].format(**fields)
    else:
        files = e.get(kind)
        if not isinstance(files, dict) or gender not in files:
            raise KeyError(f"run {run_id!r} has no {kind!r} file for {gender!r}")
        rel = files[gender]
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(f"{rel} (run {run_id!r}) is missing; restore the run files with "
                                "scripts/setup/bootstrap_post.sh")
    return p


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _record(run_id: str, gender, kind: str, p: Path) -> None:
    rec = {"run_id": run_id, "gender": gender, "kind": kind, "path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
    if rec not in _ACCESSED:
        _ACCESSED.append(rec)


def load(run_id: str, gender: str | None = None, kind: str = "metrics", **fields):
    """Parsed JSON of one run file (records it for _meta)."""
    p = path(run_id, gender, kind, **fields)
    _record(run_id, gender, kind, p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def open_path(run_id: str, gender: str | None = None, kind: str = "metrics", **fields) -> Path:
    """The path of a non-JSON run file (e.g. .npy), recorded for _meta like a load."""
    p = path(run_id, gender, kind, **fields)
    _record(run_id, gender, kind, p)
    return p


def accessed() -> list[dict]:
    return list(_ACCESSED)
