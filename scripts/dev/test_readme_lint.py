"""scripts/dev/readme_lint.py: the readability measures count what a reader sees outside the folds.

bridge-jspace's tests/test_readme_lint.py, plus the caption change. Run: uvx pytest scripts/dev/test_readme_lint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_lint as R  # noqa: E402


def measure(tmp_path, text: str) -> dict:
    p = tmp_path / "README.md"
    p.write_text(text, encoding="utf-8")
    return R.measure(p)


def test_a_ledger_label_counts_in_prose_but_not_as_link_text(tmp_path):
    assert measure(tmp_path, "H1 held on the 27B.\n")["labels"] == 1
    assert measure(tmp_path, "Held ([ledger: H1](docs/plans/hypotheses.md#h1)).\n")["labels"] == 0


def test_folds_comments_and_tables_stay_out_of_the_outside_counts(tmp_path):
    m = measure(tmp_path, "One: two.<!-- src: a | b: c -->\n\n<details>\n\nC2 held: yes.\n\n</details>\n\n| a: | b |\n")
    assert m["labels"] == 0
    assert m["colons_outside"] == 1


def test_a_per_model_slash_list_is_counted(tmp_path):
    assert measure(tmp_path, "Recall drops 0.50 / 0.74 / 0.75 ○ beyond random.\n")["slash_outside"] == 1
    assert measure(tmp_path, "Recall and lookup answer from memory / from the prompt.\n")["slash_outside"] == 0


def test_a_hard_wrapped_caption_is_measured_whole(tmp_path):
    one_line = "![a](f.png)\n\n*one two three four five six.*\n"
    wrapped = "![a](f.png)\n\n*one two three\nfour five six.*\n"
    assert measure(tmp_path, one_line)["caption_max"] == 6
    assert measure(tmp_path, wrapped)["caption_max"] == 6


def test_text_after_the_caption_paragraph_is_not_caption(tmp_path):
    assert measure(tmp_path, "![a](f.png)\n*one two.*\n\nNext paragraph here.\n")["caption_max"] == 2
    assert measure(tmp_path, "![a](f.png)\n\nNot a caption.\n")["caption_max"] == 0
