"""scripts/dev/check_sources.py on small synthetic READMEs. Run: uvx pytest -q scripts/dev/test_check_sources.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sources as C  # noqa: E402

DATA = {
    "a": 68.05,
    "b": 7.799999999999997,
    "c": -0.0547,
    "frac": 0.98989898989899,
    "n": 20000,
    "ranks": [57, 54, 65, 106],
    "rows": [{"avg": 60.25000000000001, "ci95": [57.1, 63.7]}],
    "dotted": {"I/O + SAE Desc.": {"mean": 86.95}},
    "_meta": {"x": 1.5},
}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "d.json").write_text(json.dumps(DATA), encoding="utf-8")
    (tmp_path / "METRICS.md").write_text("| I/O + SAE Desc.   | 1.43 ± 0.95 |   86.95 ± 0.94 |\n", encoding="utf-8")
    return tmp_path


def problems(text: str, root: Path) -> list[str]:
    return C.check(text, root)["problems"]


def src(key: str, path: str = "d.json") -> str:
    return f"<!-- src: {path} | {key} -->"


def test_value_matches_at_the_printed_rounding(root):
    for printed in ("68.05%", "68.1%", "68%", "68.05"):
        assert not problems(f"x {printed}{src('a')}\n", root), printed
    assert not problems(f"x 7.80{src('b')} and 7.8{src('b')} points\n", root)
    assert not problems(f"x 60.25%{src('rows[0].avg')} [57.10{src('rows[0].ci95[0]')}, 63.7{src('rows[0].ci95[1]')}]\n", root)


def test_a_mismatch_is_caught(root):
    assert problems(f"x 68.2%{src('a')}\n", root)
    assert problems(f"x 67%{src('a')}\n", root)
    assert problems(f"x 7.81{src('b')}\n", root)


def test_signs_must_match_unicode_minus_included(root):
    assert not problems(f"x −0.0547{src('c')}\n", root)
    assert not problems(f"x -0.055{src('c')}\n", root)
    assert problems(f"x 0.0547{src('c')}\n", root)
    assert problems(f"x +0.0547{src('c')}\n", root)
    assert problems(f"x −68.05{src('a')}\n", root)
    assert not problems(f"x ±0.055{src('c')}\n", root)


def test_transforms(root):
    assert not problems(f"x 98.99%{src('frac ×100')}\n", root)
    assert problems(f"x 98.99%{src('frac')}\n", root)
    assert not problems(f"within 0.055{src('c abs')} of zero\n", root)
    assert not problems(f"x 61{src('ranks median')}, n=4{src('ranks len')}, 54{src('ranks min')}\n", root)
    assert problems(f"x 61{src('a median')}\n", root)  # a reduction needs a list
    assert problems(f"x 61{src('ranks mean')}\n", root)  # unknown transform


def test_bracketed_keys_and_thousands_separators(root):
    key = 'dotted["I/O + SAE Desc."].mean'
    assert not problems(f"x 87%{src(key)}\n", root)
    assert not problems(f"B=20,000{src('n')}\n", root)


def test_table_cells_use_an_escaped_separator(root):
    row = "| method | 68.05%<!-- src: d.json \\| a --> |"
    text = f"| method | avg |\n| --- | --- |\n{row}\n"
    assert not problems(text, root)
    assert C.check(text, root)["sourced"] == ["l.3: 68.05%"]


def test_unsourced_numbers_are_caught(root):
    assert problems("- the gap is 0.53 on direct\n", root)
    assert problems(f"- 7.8{src('b')} points, then 0.53 more\n", root)  # one comment per number
    assert problems("- the auditor says female 90% of the time\n", root)


def test_names_are_exempt(root):
    text = (
        "Gemma 2 9B and Gemma 3 4B at layer 11 instead of 23, layers 9, 10 and 12, layer-11 readout (Finding 2, MATS 12.0,\n"
        "August 2026, Cywiński et al. (2025), n=1000, top-10 tokens of 200, L11, float32, CC BY 4.0, arXiv v2). Layer\n"
        "11 was chosen. Code: `top_k=200`.\n\n```bash\nmake numbers  # 9 seconds\n```\n\n1. **A step.**\n"
    )
    assert not problems(text, root)


def test_header_rows_and_layer_columns_are_exempt(root):
    text = f"| method | layer | avg |\n| --- | --- | --- |\n| Acts Tokens | 23 | 60.25%<!-- src: d.json \\| rows[0].avg --> |\n"
    assert not problems(text, root)
    assert problems("| method | layer | avg |\n| --- | --- | --- |\n| Acts Tokens | 23 | 60.25% |\n", root)


def test_a_digit_that_is_neither_number_nor_name_is_caught(root):
    assert problems("run on an A10G\n", root)


def test_broken_sources_are_caught(root):
    assert problems(f"x 68%{src('a', 'no/such.json')}\n", root)
    assert problems(f"x 68%{src('nosuchkey')}\n", root)
    assert problems(f"x 68%{src('rows[3].avg')}\n", root)
    assert problems(f"x 1.5{src('_meta.x')}\n", root)
    assert problems(f"x 68%{src('rows')}\n", root)  # not a number
    assert problems(f"no number here{src('a')}\n", root)


def test_markdown_sources_use_a_substring_key(root):
    assert not problems(f"x 87%{src('I/O + SAE Desc. | 1.43 ± 0.95 | 86.95', 'METRICS.md')}\n", root)
    assert problems(f"x 88%{src('I/O + SAE Desc. | 1.43 ± 0.95 | 86.95', 'METRICS.md')}\n", root)
    assert problems(f"x 87%{src('I/O + SAE Desc. | 86.95', 'METRICS.md')}\n", root)  # not a substring


def test_src_none_is_accepted_and_reported(root):
    r = C.check("about 6<!-- src: none | wall-clock time --> seconds\n", root)
    assert not r["problems"] and r["none"] == ["l.1: 6 (wall-clock time)"]


def test_missing_link_targets_are_caught(root):
    assert problems("![a](figures/nope.png)\n", root)
    assert problems("see [the ledger](NOPE.md#x)\n", root)
    assert not problems("see [the table](METRICS.md#x), [here](#top) and [web](https://example.org/1)\n", root)


def test_every_source_in_the_real_readme_resolves_and_matches():
    """Unsourced numbers are reported by the CLI; a src comment that is present must be right."""
    readme = C.REPO / "README.md"
    bad = [p for p in C.check(readme.read_text(encoding="utf-8"))["problems"] if "has no src comment" not in p]
    assert not bad, bad


def test_a_forbidden_file_is_refused(root, monkeypatch):
    monkeypatch.setitem(C.FORBIDDEN, "derived/user_gender/x.json", "embargoed")
    assert problems("x 68%<!-- src: derived/user_gender/x.json | a -->\n", root)


def test_src_none_needs_a_reason(root):
    assert problems("x 25<!-- src: none | -->\n", root)
    assert problems("x 25<!-- src: none | effort -->\n", root)
    assert not problems("x 25<!-- src: none | hours spent on the project -->\n", root)
