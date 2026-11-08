"""Check that every number in README.md carries a source, and that the source holds that number as printed.

    python scripts/dev/check_sources.py [README.md]

Prints a summary and exits 1 on any problem: a missing source file, a key that does not resolve, a value that does not
match the printed number at its printed rounding, a source comment that does not follow a number, a link or image path
that does not exist, or a number with no source comment.

Comment syntax (bridge-jspace's tests/test_readme_sources.py format)
    68.05%<!-- src: derived/user_gender/reproduction_bar_chart_direct.json | rows[3].avg -->
The comment sits directly after the number it sources, after a trailing "%", "pp" or "×". In a table row the separator
is written "\\|" (a bare pipe would split the cell). One comment per number.

Key syntax, JSON source
    rows[3].avg  pairwise[0].ci95[1]  point_accuracies.acts_tokens_l11      dotted keys and [i] list indices
    user_gender["Direct Secret-keeping model"]["I/O + SAE Desc."].mean      ["..."] for keys with dots or spaces
The `_meta` block is provenance, not a source. Transforms may follow the key, space-separated, applied left to right:
    ×100                      a stored fraction printed as a percent
    abs                       the magnitude ("within 0.055 of zero", "±0.055")
    median, min, max, len     reduce a stored list of numbers to one
Key syntax, any other source (e.g. METRICS.md, which holds the paper's tables): a substring of the file, compared with
whitespace collapsed and HTML comments removed; the value is the last number in the key.

    <!-- src: none | <reason> -->
marks a number no derived file holds and that is not a result (e.g. a wall-clock time). It is accepted, but every use is
printed in the summary.

Value match. The printed number's decimals d set the rounding: |value − printed| ≤ ½·10^−d, in decimal arithmetic on the
value's shortest repr (so 7.799999999999997 matches "7.80" and "7.8", 96.97 does not match "96"). Signs must match: a
printed "−" (U+2212) or "-" needs a negative value, "+" or no sign a non-negative one; "±" compares magnitudes.
Thousands separators ("20,000") are allowed.

What counts as a number. Every digit run outside fenced code, code spans, headings, image lines, HTML comments and tags,
link targets, ordered-list markers, table header rows, table columns headed "layer", and the NAMES patterns below (digits that name things
rather than measure them: layers, model sizes, years, licence and programme versions, design counts). A digit that is
neither a NAMES match nor part of a well-formed number is reported too, so a new kind of name must be added to NAMES.
"""

from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN: dict[str, str] = {}  # derived files the README must not cite, with the reason (empty since P1 was read)

SRC = re.compile(r"<!--\s*src:\s*([^|\\]*?)\s*\\?\|\s*(.*?)\s*-->", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)
NUMBER = re.compile(r"(?<![\w.])([+\-−±]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|pp(?![A-Za-z])|×)?(?![\w])")
NUM_BEFORE = re.compile(r"(?<![\w.])([+\-−±]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(%|pp|×)?$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
PATH_TOKEN = re.compile(r'\.?([A-Za-z_][\w-]*)|\[(\d+)\]|\["([^"]*)"\]')
TRANSFORMS = {"×100", "abs", "median", "min", "max", "len"}

# Digits that are names, not quantities. Matched on the whole page, so "\s+" also spans a hard line wrap.
NAMES = [
    r"\b\d+(?:\.\d+)?B\b",  # model sizes: 9B, 4B, 70B
    r"\bGemma[- ]\d(?:-\d+[bB](?:-it)?)?\b",  # Gemma 2, Gemma 3, Gemma-2-9b-it
    r"\bGPT-\d(?:\.\d)?(?:-mini)?\b",
    r"\bA100 \d+GB\b|\bApple M\d\b",  # hardware
    r"\b(?:bf16|float32|sha256)\b",
    r"\bCC BY \d\.\d\b|\bMATS \d+\.\d\b|\bv\d+(?:\.\d+)?\b|\bAppendix [A-Z]\.\d\b",  # licence, programme, versions
    r"\bFinding \d\b|\blevel \d\b|\bL\d+\b",  # sections, verification levels, layer tags
    r"\b[Ll]ayers?(?:-|\s+)\d+(?:\s*(?:–|-|,|and|or|of|instead of)\s*\d+)*",  # layer 11, layers 9, 10 and 12, layer 11 of 42
    r"\(\d+ instead of \d+\)|\b\d+–\d+ window\b|\bput \d+ in it\b",  # layer ids without the word "layer"
    r"\((?:c\) )?20\d\d\)|\(c\) 20\d\d\b|\bet al\. 20\d\d\b",  # years
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d\b",
    r"\btop-\d+\b|\b(?:of|all) 200\b",  # the evidence list's length
    r"\bn=(?:1000|2000)\b|\b100 prompts(?: \d+ times)?\b",  # the design: 100 prompts x 10 responses per checkpoint
    r"\b\d+-item pilot\b|\b\d+ items, \d+ prompts × \d+ responses\b",  # the layer pilot's design
    r"\b\d+ pairs each\b|\b95% (?:intervals|CI)\b",  # contrastive pairs per control; the interval level
    r"`BATCH_SIZE` \d+ → \d+",  # an upstream config edit
]


# ---- source resolution ------------------------------------------------------------------------------------------------


def parse_key(key: str) -> tuple[list[str | int], list[str]]:
    """'rows[3].avg ×100' -> (['rows', 3, 'avg'], ['×100']). Raises ValueError on a malformed key."""
    path, i = [], 0
    while i < len(key) and not key[i].isspace():
        m = PATH_TOKEN.match(key, i)
        if not m or (m.group(1) is not None and i > 0 and key[i] != "."):
            raise ValueError(f"malformed key at {key[i:]!r}")
        path.append(m.group(1) if m.group(1) is not None else int(m.group(2)) if m.group(2) is not None else m.group(3))
        i = m.end()
    transforms = key[i:].split()
    if bad := [t for t in transforms if t not in TRANSFORMS]:
        raise ValueError(f"unknown transform {bad[0]!r}")
    if not path:
        raise ValueError("empty key")
    return path, transforms


def apply(value, transforms: list[str]):
    for t in transforms:
        if t in {"median", "min", "max", "len"}:
            if not isinstance(value, list) or not value:
                raise ValueError(f"{t} needs a non-empty list")
            if t == "len":
                value = len(value)
            elif t == "median":
                s, n = sorted(value), len(value)
                value = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
            else:
                value = min(value) if t == "min" else max(value)
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{t} needs a number")
        else:
            value = value * 100 if t == "×100" else abs(value)
    return value


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def resolve(root: Path, path: str, key: str) -> tuple[float | None, str | None]:
    """(value, None) if the key resolves to a number, else (None, the reason)."""
    if path in FORBIDDEN:
        return None, f"not citable: {FORBIDDEN[path]}"
    p = root / path
    if not p.is_file():
        return None, "no such file"
    if p.suffix != ".json":
        page = collapse(COMMENT.sub(" ", p.read_text(encoding="utf-8")))
        if collapse(key) not in page:
            return None, "key text not found"
        nums = list(NUMBER.finditer(key))
        if not nums:
            return None, "key text holds no number"
        sign, whole, frac, _ = nums[-1].groups()
        v = float(whole.replace(",", "") + (frac or ""))
        return (-v if sign in ("-", "−") else v), None
    try:
        steps, transforms = parse_key(key)
    except ValueError as e:
        return None, str(e)
    if steps[0] == "_meta":
        return None, "_meta is provenance, not a source"
    node = json.loads(p.read_text(encoding="utf-8"))
    for s in steps:
        try:
            node = node[s]
        except (KeyError, IndexError, TypeError):
            return None, f"no {s!r} at this point of the path"
    try:
        node = apply(node, transforms)
    except ValueError as e:
        return None, str(e)
    if not isinstance(node, (int, float)) or isinstance(node, bool):
        return None, f"resolves to a {type(node).__name__}, not a number"
    return float(node), None


def matches(value: float, sign: str, whole: str, frac: str | None) -> bool:
    """Whether `value` prints as the number (sign, whole, frac) at its printed rounding."""
    printed = Decimal(whole.replace(",", "") + (frac or ""))
    d = len(frac) - 1 if frac else 0
    v = Decimal(repr(value))
    if sign == "±":
        v = abs(v)
    elif sign in ("-", "−"):
        printed = -printed
    if abs(v - printed) > Decimal(5) / Decimal(10) ** (d + 1):
        return False
    rounded_nonzero = round(abs(v), d) != 0
    if sign in ("-", "−"):
        return v < 0 or not rounded_nonzero
    return v >= 0 or not rounded_nonzero


# ---- the page ---------------------------------------------------------------------------------------------------------


def blank(text: str, start: int, end: int) -> str:
    """text with [start, end) replaced by spaces, newlines kept (offsets stay valid)."""
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def blank_all(text: str, pattern: str, flags: int = 0, group: int = 0) -> str:
    for m in reversed(list(re.finditer(pattern, text, flags))):
        text = blank(text, m.start(group), m.end(group))
    return text


def cells(line: str) -> list[tuple[int, int]]:
    """(start, end) of each cell of a table row; "\\|" inside a source comment does not split."""
    bounds = [m.start() for m in re.finditer(r"(?<!\\)\|", line)]
    return list(zip(bounds, bounds[1:]))


def mask_tables(text: str) -> str:
    """Blank table header rows and the cells of columns headed "layer"."""
    lines = text.split("\n")
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)
    layer_cols: set[int] = set()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith("|"):
            layer_cols = set()
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if re.fullmatch(r"\|[-:| ]+\|", nxt):  # a header row
            heads = [re.sub(r"[*`\s]", "", ln[a + 1 : b]).lower() for a, b in cells(ln)]
            layer_cols = {j for j, h in enumerate(heads) if h == "layer"}
            text = blank(text, offsets[i], offsets[i] + len(ln))
            continue
        for j, (a, b) in enumerate(cells(ln)):
            if j in layer_cols:
                text = blank(text, offsets[i] + a + 1, offsets[i] + b)
    return text


def masked(text: str) -> str:
    """The page with everything that is not a checkable number blanked out, offsets unchanged."""
    t = blank_all(text, r"^```.*?^```", re.S | re.M)  # fenced code
    t = blank_all(t, r"<!--.*?-->", re.S)
    t = blank_all(t, r"^[ \t]*(?:#|!\[).*$", re.M)  # headings, image lines
    t = blank_all(t, r"^[ \t]*\d+\.(?=[ \t])", re.M)  # ordered-list markers
    t = blank_all(t, r"\]\(([^)]*)\)", group=1)  # link targets
    t = blank_all(t, r"<[a-zA-Z/][^>]*>")  # HTML tags (<details>, <a id=...>)
    for pat in NAMES:
        t = blank_all(t, pat)
    t = blank_all(t, r"`[^`\n]*`")  # code spans
    return mask_tables(t)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def check(text: str, root: Path = REPO) -> dict:
    """{'sourced': [...], 'none': [...], 'problems': [...]} for the page `text`, sources resolved under `root`."""
    sourced, none, problems = [], [], []
    for m in SRC.finditer(text):
        path, key, ln = m.group(1).strip(), m.group(2).strip(), line_of(text, m.start())
        before = NUM_BEFORE.search(text[max(0, m.start() - 40) : m.start()])
        if not before:
            problems.append(f"l.{ln}: src comment does not follow a number ({path} | {key})")
            continue
        sign, whole, frac, _ = before.groups()
        number = before.group(0)
        if path == "none":
            if len(key.split()) < 2:  # a reason is required: "src: none | <reason>"
                problems.append(f"l.{ln}: {number}: src: none needs a reason")
            else:
                none.append(f"l.{ln}: {number} ({key})")
            continue
        value, why = resolve(root, path, key)
        if why:
            problems.append(f"l.{ln}: {number}: {path} | {key}: {why}")
        elif not matches(value, sign, whole, frac):
            problems.append(f"l.{ln}: {number} does not match {path} | {key} = {value!r}")
        else:
            sourced.append(f"l.{ln}: {number}")
    t = masked(text)
    for m in NUMBER.finditer(t):
        if not text.startswith("<!--", m.end()) or not SRC.match(text, m.end()):
            problems.append(f"l.{line_of(text, m.start())}: {m.group(0)} has no src comment")
    t = blank_all(t, NUMBER.pattern)
    for m in re.finditer(r"\S*\d\S*", t):
        problems.append(f"l.{line_of(text, m.start())}: {m.group(0)!r} is neither a number nor a NAMES match")
    for target in LINK.findall(COMMENT.sub("", text)):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        if not (root / target.split("#", 1)[0]).exists():
            problems.append(f"link target missing: {target}")
    return {"sourced": sourced, "none": none, "problems": problems}


def main(argv: list[str]) -> int:
    readme = Path(argv[0]) if argv else REPO / "README.md"
    if not readme.is_absolute():
        readme = Path.cwd() / readme
    r = check(readme.read_text(encoding="utf-8"), readme.resolve().parent)
    print(f"{readme.name}: {len(r['sourced'])} numbers sourced, {len(r['none'])} marked src: none, "
          f"{len(r['problems'])} problems")  # fmt: skip
    for n in r["none"]:
        print(f"  none     {n}")
    for p in r["problems"]:
        print(f"  PROBLEM  {p}")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
