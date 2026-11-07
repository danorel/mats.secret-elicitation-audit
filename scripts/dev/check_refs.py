#!/usr/bin/env python3
"""Every image under figures/ (outside _ folders) is referenced from README.md or EXPERIMENTS.md, and every image
those two files reference exists. Run by `make check`; exit 1 on any mismatch."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "EXPERIMENTS.md"]
IMAGE = re.compile(r"(figures/[\w./-]+\.(?:png|svg|jpg))")


def main() -> int:
    refs = set()
    for doc in DOCS:
        refs |= set(IMAGE.findall((ROOT / doc).read_text(encoding="utf-8")))
    images = {str(p.relative_to(ROOT)) for p in (ROOT / "figures").rglob("*")
              if p.suffix in (".png", ".svg", ".jpg") and not any(part.startswith("_") for part in p.parts)}
    problems = [f"not referenced from {' or '.join(DOCS)}: {p}" for p in sorted(images - refs)]
    problems += [f"referenced but missing: {p}" for p in sorted(refs - images)]
    print("\n".join(problems) if problems else f"check_refs: {len(images)} images, all referenced, none missing")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
