"""scripts/dev/check_refs.py: links and kept anchors. Run: uvx pytest scripts/dev/test_check_refs.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_refs as C  # noqa: E402


def test_slug_follows_github():
    assert C.slug("Finding 3. The checkpoints sit apart <a id=\"x\"></a>") == "finding-3-the-checkpoints-sit-apart"
    assert C.slug("What I predicted, and what happened") == "what-i-predicted-and-what-happened"
    assert C.slug("Finding 4. The auditor's default") == "finding-4-the-auditors-default"
    assert C.slug("8. Shared vocabulary") == "8-shared-vocabulary"


def test_anchors_include_explicit_ids_and_skip_code_fences(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n\n<a id=\"kept\"></a>\n\n```\n# not a heading\n```\n\n## Title\n", encoding="utf-8")
    assert C.anchors(p) == {"title", "title-1", "kept"}


def test_a_missing_anchor_and_a_missing_file_are_caught(tmp_path, monkeypatch):
    (tmp_path / "b.md").write_text("## Here\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("[ok](b.md#here) [bad](b.md#gone) [nofile](c.md) [web](https://x.org/#y)\n",
                                   encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "PAGES", ["a.md"])
    problems, n = C.link_problems()
    assert n == 3
    assert problems == ["a.md: no anchor #gone in b.md", "a.md: link to a missing file: c.md"]


def test_a_kept_anchor_that_disappears_is_caught(tmp_path, monkeypatch):
    (tmp_path / "R.md").write_text("## New name <a id=\"old-name\"></a>\n", encoding="utf-8")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(C, "KEPT", {"R.md": ["old-name", "new-name", "vanished"]})
    assert C.kept_problems() == ["R.md: kept anchor #vanished is gone"]
