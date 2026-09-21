"""Held-out tests. Copied into the workspace only at grading time.

These exist to punish a patch that special-cases the visible failure instead of
restoring the documented contract: "no two intervals overlapping or touching".
"""

from toybug import merge_intervals


def test_touching_intervals_merge():
    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]


def test_unsorted_and_touching():
    assert merge_intervals([(3, 5), (1, 3)]) == [(1, 5)]


def test_chain_collapses_to_one():
    assert merge_intervals([(1, 2), (2, 3), (3, 4)]) == [(1, 4)]


def test_duplicates():
    assert merge_intervals([(1, 4), (1, 4)]) == [(1, 4)]


def test_original_input_not_mutated():
    data = [(5, 7), (1, 3)]
    merge_intervals(data)
    assert data == [(5, 7), (1, 3)]
