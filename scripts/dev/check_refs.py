#!/usr/bin/env python3
"""Every image under figures/readme/ is referenced from README.md or EXPERIMENTS.md, every image those two files
reference exists and is under figures/readme/, and figures/writeup/ (the write-up's copies, referenced from outside
the repo) holds exactly the same file names as figures/readme/. Run by `make check`; exit 1 on any mismatch."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "EXPERIMENTS.md"]
LINKED, MIRROR = "figures/readme", "figures/writeup"
SUFFIXES = (".png", ".svg", ".jpg")
IMAGE = re.compile(r"(figures/[\w./-]+\.(?:png|svg|jpg))")


def images(folder: str) -> set[str]:
    return {str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*") if p.suffix in SUFFIXES}


def main() -> int:
    refs = set()
    for doc in DOCS:
        refs |= set(IMAGE.findall((ROOT / doc).read_text(encoding="utf-8")))
    linked, mirror = images(LINKED), images(MIRROR)
    def names(paths: set[str], folder: str) -> set[str]:
        return {p.removeprefix(folder + "/") for p in paths}

    problems =[f"not referenced from {' or '.join(DOCS)}: {p}" for p in sorted(linked - refs)]
    for p in sorted(refs - linked):
        problems.append(f"referenced outside {LINKED}/: {p}" if (ROOT / p).exists() else f"referenced but missing: {p}")
    problems += [f"in {LINKED}/ but not {MIRROR}/: {n}" for n in sorted(names(linked, LINKED) - names(mirror, MIRROR))]
    problems += [f"in {MIRROR}/ but not {LINKED}/: {n}" for n in sorted(names(mirror, MIRROR) - names(linked, LINKED))]
    print("\n".join(problems) if problems else
          f"check_refs: {len(linked)} images in {LINKED}/, all referenced, none missing; {MIRROR}/ mirrors it")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
