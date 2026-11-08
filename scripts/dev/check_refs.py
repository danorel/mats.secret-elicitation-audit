#!/usr/bin/env python3
"""Links, anchors and figure folders. Run by `make check`; exit 1 on any problem.

- Images: every image under figures/readme/ is referenced from README.md or EXPERIMENTS.md, every image those two
  files reference exists and is under figures/readme/, and figures/writeup/ (the write-up's copies, referenced from
  outside the repo) holds exactly the same file names as figures/readme/.
- Links: every relative link in the repo's reader-facing Markdown pages (PAGES) resolves to an existing file, and a
  `#fragment` to a heading or `<a id>` in the target page (GitHub's heading slugs).
- Kept anchors: anchors that outside documents link to (the write-up, the HF card) or that were kept when a heading
  was renamed (KEPT) must still exist.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "EXPERIMENTS.md"]
LINKED, MIRROR = "figures/readme", "figures/writeup"
SUFFIXES = (".png", ".svg", ".jpg")
IMAGE = re.compile(r"(figures/[\w./-]+\.(?:png|svg|jpg))")
PAGES = ["README.md", "EXPERIMENTS.md", "DEVELOPER.md", "AGENTS.md", "CHANGELOG.md", "METRICS.md",
         "docs/examples.md", "docs/STANDARD.md", "figures/readme/README.md", "figures/writeup/README.md"]
# The README's anchors kept through the docs re-cut (old heading slugs now held by <a id>), plus the headings the
# write-up links to; and every ledger card's anchor (the write-up's Appendix C links into EXPERIMENTS.md).
KEPT = {
    "README.md": ["summary", "background", "how-the-benchmark-works", "key-terms", "what-i-did", "findings",
                  "finding-1-the-gain-comes-from-the-read-layer-not-from-the-direction",
                  "finding-2-fine-tuning-wrote-the-secret-onto-the-base-models-own-gender-axis",
                  "finding-3-the-secret-does-not-depend-on-the-conversation",
                  "finding-4-the-auditors-default-follows-the-evidence-fields-shape-not-its-content",
                  "more-results", "where-it-does-not-hold", "why-this-matters", "open-questions-and-next-steps",
                  "how-i-checked", "related-work-and-what-is-new", "repository-guide"],
    "EXPERIMENTS.md": ["reproduction", "layer-choice", "outside-selection", "read-layer", "precision", "delta-alignment", "constant-offset",
                       "position-control", "ordinary-chat", "per-prompt", "nearest-tokens", "patchscope", "names",
                       "blank-evidence", "acts-desc", "resampling", "scoped", "how-to-read"],
}
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)|<(?:img|a)\s[^>]*?(?:src|href)=\"([^\"]+)\"")
FENCE = re.compile(r"```.*?```", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)


def images(folder: str) -> set[str]:
    return {str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*") if p.suffix in SUFFIXES}


def slug(heading: str) -> str:
    """GitHub's heading anchor: tags and inline-code ticks dropped, lowercased, punctuation removed, spaces to '-'."""
    text = re.sub(r"<[^>]+>", "", heading).replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(page: Path) -> set[str]:
    text = FENCE.sub("", page.read_text(encoding="utf-8"))
    out, seen = set(re.findall(r"<a id=\"([^\"]+)\"", text)), {}
    for h in re.findall(r"^#{1,6} (.+?)\s*$", text, flags=re.M):
        s = slug(h)
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def image_problems() -> list[str]:
    refs = set()
    for doc in DOCS:
        refs |= set(IMAGE.findall((ROOT / doc).read_text(encoding="utf-8")))
    linked, mirror = images(LINKED), images(MIRROR)

    def names(paths: set[str], folder: str) -> set[str]:
        return {p.removeprefix(folder + "/") for p in paths}

    problems = [f"not referenced from {' or '.join(DOCS)}: {p}" for p in sorted(linked - refs)]
    for p in sorted(refs - linked):
        problems.append(f"referenced outside {LINKED}/: {p}" if (ROOT / p).exists() else f"referenced but missing: {p}")
    problems += [f"in {LINKED}/ but not {MIRROR}/: {n}" for n in sorted(names(linked, LINKED) - names(mirror, MIRROR))]
    problems += [f"in {MIRROR}/ but not {LINKED}/: {n}" for n in sorted(names(mirror, MIRROR) - names(linked, LINKED))]
    return problems


def link_problems() -> tuple[list[str], int]:
    problems, n = [], 0
    cache: dict[Path, set[str]] = {}
    for rel in PAGES:
        page = ROOT / rel
        if not page.is_file():
            problems.append(f"page missing: {rel}")
            continue
        text = COMMENT.sub("", FENCE.sub("", page.read_text(encoding="utf-8")))
        for m in LINK.finditer(text):
            target = m.group(1) or m.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
                continue
            n += 1
            path, _, frag = target.partition("#")
            dest = (page.parent / path).resolve() if path else page
            if not dest.exists():
                problems.append(f"{rel}: link to a missing file: {target}")
                continue
            if frag and dest.suffix == ".md":
                if dest not in cache:
                    cache[dest] = anchors(dest)
                if frag not in cache[dest]:
                    problems.append(f"{rel}: no anchor #{frag} in {dest.relative_to(ROOT)}")
    return problems, n


def kept_problems() -> list[str]:
    problems = []
    for rel, wanted in KEPT.items():
        have = anchors(ROOT / rel)
        problems += [f"{rel}: kept anchor #{a} is gone" for a in wanted if a not in have]
    return problems


def main() -> int:
    links, n_links = link_problems()
    problems = image_problems() + links + kept_problems()
    n_kept = sum(len(v) for v in KEPT.values())
    print("\n".join(problems) if problems else
          f"check_refs: {len(images(LINKED))} images in {LINKED}/, all referenced, none missing; {MIRROR}/ mirrors it; "
          f"{n_links} links in {len(PAGES)} pages resolve; {n_kept} kept anchors present")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
