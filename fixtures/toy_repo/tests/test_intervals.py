from toybug import merge_intervals


def test_empty():
    assert merge_intervals([]) == []


def test_single():
    assert merge_intervals([(1, 4)]) == [(1, 4)]


def test_nested_interval_is_absorbed():
    assert merge_intervals([(1, 10), (2, 5)]) == [(1, 10)]


def test_sorted_overlapping():
    assert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]


def test_unsorted_input():
    # Reported failure: callers do not always pass intervals in order.
    assert merge_intervals([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]
