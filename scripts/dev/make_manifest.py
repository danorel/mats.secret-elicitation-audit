#!/usr/bin/env python3
"""manifest.csv for the HF dataset danorel/mats.esk-audit, from a local copy of the dataset plus runs.toml.

One row per file: path, size, sha256, run_id, canonical, produced_by, notes.
- run_id / canonical / produced_by come from runs.toml when the file is one a registered run names (or one of a
  run's listed `candidates`, then canonical = "no (candidate)").
- Otherwise produced_by is the upstream script the path's layout implies (inference/, baseline/), or
  "unknown (file dated <date>: session N)" from the date in the file name and the session logs' dates. Nothing else
  is inferred; provenance is never invented.

    python3 scripts/dev/make_manifest.py <local dataset dir> [--out <dir>/manifest.csv] [--exclude user_gender/plots]
"""
import argparse
import csv
import hashlib
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SESSIONS = {"20260820": "1", "20260822": "2", "20260823": "3", "20260824": "4", "20260825": "5",
            "20260826": "6 or 7", "20260827": "8"}  # from log/2026-*-session-*.md
LAYOUT = [  # (path regex, produced_by) for files no run entry names
    (r"/inference/", "user_gender/scripts/run_inference.sh"),
    (r"/baseline/logit_lens/", "user_gender/scripts/run_get_logit_lens.sh"),
    (r"/baseline/residual_tokens/", "user_gender/scripts/run_get_residual_tokens.sh"),
    (r"/baseline/sae_features", "user_gender/scripts/run_get_sae_features.sh"),
    (r"/baseline/audit/io/", "user_gender/scripts/run_auditor_input_output.sh"),
    (r"/baseline/audit/logit_lens/", "user_gender/scripts/run_auditor_logit_lens.sh"),
    (r"/baseline/audit/residual_tokens/", "user_gender/scripts/run_auditor_residual_tokens.sh"),
    (r"/baseline/audit/sae_tokens/", "user_gender/scripts/run_auditor_sae_tokens.sh"),
    (r"/baseline/audit/sae_feature_descriptions/", "user_gender/scripts/run_auditor_sae_descriptions.sh"),
    (r"/user_gender/inputs/gender_direction/(contrastive|random_control)/", "user_gender/scripts/run_get_gender_direction.sh"),
    (r"/user_gender/inputs/gender_direction/control_", "analysis/semantic_null_cosine_control.py (builds it via elicitation_methods/gender_direction.py --mode build)"),
]


def registry_index() -> dict:
    """relative path -> (run_id, canonical, produced_by, note)"""
    with (ROOT / "runs.toml").open("rb") as f:
        runs = tomllib.load(f)["runs"]
    idx = {}
    for rid, e in runs.items():
        meta = (rid, e.get("canonical", ""), e.get("produced_by", ""), e.get("note", ""))
        for v in e.values():
            if isinstance(v, dict):
                for p in v.values():
                    idx[p] = meta
        for p in e.get("candidates", []):
            idx.setdefault(p, (rid, "no (candidate)", e.get("produced_by", ""), "listed as a candidate of " + rid))
        if "template" in e:
            for var in e.get("variants", [None]):
                for layer in e.get("layers", [None]):
                    idx[e["template"].format(variant=var, layer=layer)] = meta
    return idx


def guess(rel: str):
    if rel.endswith("/README.md"):
        return "documentation: a hand-written note on the folder's runs"
    for pat, prod in LAYOUT:
        if re.search(pat, "/" + rel):
            return prod
    m = re.findall(r"(2026\d{4})_\d{6}", rel)
    if m:
        d = m[-1]
        return f"unknown (file dated {d[:4]}-{d[4:6]}-{d[6:]}: session {SESSIONS.get(d, '?')})"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("local_dir")
    ap.add_argument("--out", default=None, help="Default: <local_dir>/manifest.csv")
    ap.add_argument("--exclude", nargs="*", default=["user_gender/plots"])
    a = ap.parse_args()
    base = Path(a.local_dir)
    idx = registry_index()
    dirs = {os.path.dirname(k): v for k, v in idx.items() if not v[1].startswith("no (candidate)")}
    rows = []
    for p in sorted(base.rglob("*")):
        rel = str(p.relative_to(base))
        if not p.is_file() or rel.startswith(".cache/") or rel in ("manifest.csv", "README.md", ".gitattributes"):
            continue
        if any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in a.exclude):
            continue
        data = p.read_bytes()
        if rel in idx:
            rid, canonical, produced, note = idx[rel]
        elif os.path.dirname(rel) in dirs:  # beside a registered run's file (its guessing.json, evidence.json, ...)
            r = dirs[os.path.dirname(rel)]
            rid, canonical, produced, note = r[0], "—", r[2], f"same folder as run {r[0]}"
        else:
            rid, canonical, produced, note = "", "—", guess(rel), ""
        rows.append([rel, len(data), hashlib.sha256(data).hexdigest(), rid, canonical, produced, note])
    out = Path(a.out) if a.out else base / "manifest.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "size", "sha256", "run_id", "canonical", "produced_by", "notes"])
        w.writerows(rows)
    unknown = sum(1 for r in rows if r[5].startswith("unknown"))
    print(f"wrote {out}: {len(rows)} files, {sum(r[1] for r in rows)} bytes, {sum(1 for r in rows if r[3])} in "
          f"runs.toml, {unknown} with produced_by 'unknown'")


if __name__ == "__main__":
    main()
