"""Tests for the harness itself.

None of these call a model, so they run free and offline. They pin the two
things that are easy to break silently: the path sandbox, and the scoring
math that every later experiment is measured against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.tools import PathEscape, Toolbox
from core.types import ToolCall, Usage
from eval.grade import grade
from tasks.toy import ToyTask


# ---- sandbox -------------------------------------------------------


def test_path_escape_is_blocked(tmp_path: Path) -> None:
    tb = Toolbox(tmp_path)
    (tmp_path / "ok.txt").write_text("fine", encoding="utf-8")
    assert "fine" in tb.read_file("ok.txt")
    with pytest.raises(PathEscape):
        tb.read_file("../secrets.txt")


def test_tool_errors_come_back_as_results_not_exceptions(tmp_path: Path) -> None:
    tb = Toolbox(tmp_path)
    result = tb.execute(ToolCall(id="1", name="read_file", args={"path": "nope.txt"}))
    assert result.is_error
    assert "nope.txt" in result.content


def test_edit_file_refuses_ambiguous_match(tmp_path: Path) -> None:
    tb = Toolbox(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = tb.execute(
        ToolCall(id="1", name="edit_file", args={"path": "a.py", "old": "x = 1", "new": "x = 2"})
    )
    assert result.is_error
    assert "matches 2 times" in result.content


# ---- usage / cost --------------------------------------------------


def test_usage_adds() -> None:
    total = Usage(input_tokens=10, output_tokens=5) + Usage(input_tokens=1, output_tokens=2)
    assert total.input_tokens == 11
    assert total.output_tokens == 7


# ---- scoring -------------------------------------------------------


def _prepare(tmp_path: Path):
    return ToyTask().prepare("toy-intervals", tmp_path / "ws")


def test_untouched_repo_scores_46(tmp_path: Path) -> None:
    """Doing nothing still banks the regression weight and 2 incidental hidden tests."""
    card = grade(_prepare(tmp_path))
    assert not card.target.all_green
    assert card.regression.all_green
    assert (card.hidden.passed, card.hidden.total) == (2, 5)
    assert card.score == 46
    assert not card.resolved


def test_lazy_fix_scores_76_and_is_not_resolved(tmp_path: Path) -> None:
    """Sorting the input satisfies every visible test and still loses the hidden ones.

    This is the overfit trap the whole design exists to defeat: the agent sees
    a fully green suite and is still 24 points short.
    """
    inst = _prepare(tmp_path)
    src = inst.workspace / "toybug" / "intervals.py"
    src.write_text(
        src.read_text(encoding="utf-8").replace(
            "    merged = [list(intervals[0])]",
            "    intervals = sorted(intervals)\n    merged = [list(intervals[0])]",
        ),
        encoding="utf-8",
    )
    card = grade(inst)
    assert card.target.all_green
    assert card.regression.all_green
    assert (card.hidden.passed, card.hidden.total) == (2, 5)
    assert card.score == 76
    assert not card.resolved


def test_correct_fix_scores_100(tmp_path: Path) -> None:
    """Honouring the documented contract -- sorted *and* touching intervals merged."""
    inst = _prepare(tmp_path)
    src = inst.workspace / "toybug" / "intervals.py"
    src.write_text(
        src.read_text(encoding="utf-8")
        .replace(
            "    merged = [list(intervals[0])]",
            "    intervals = sorted(intervals)\n    merged = [list(intervals[0])]",
        )
        .replace("if start < last[1]:", "if start <= last[1]:"),
        encoding="utf-8",
    )
    card = grade(inst)
    assert card.score == 100
    assert card.resolved


def test_cheating_by_editing_the_test_does_not_reach_100(tmp_path: Path) -> None:
    """Rewriting the failing test buys the target weight and nothing else."""
    inst = _prepare(tmp_path)
    test_file = inst.workspace / "tests" / "test_intervals.py"
    test_file.write_text(
        test_file.read_text(encoding="utf-8").replace(
            "assert merge_intervals([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]",
            "assert merge_intervals([(5, 7), (1, 3), (2, 4)]) == [(5, 7)]",
        ),
        encoding="utf-8",
    )
    card = grade(inst)
    assert card.target.all_green  # the doctored test passes
    assert card.score == 76  # but hidden tests are untouched, so it caps out
    assert not card.resolved
