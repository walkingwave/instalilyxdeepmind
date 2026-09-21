"""Interval utilities."""


def merge_intervals(intervals):
    """Merge overlapping intervals into the smallest equivalent set.

    Args:
        intervals: iterable of (start, end) pairs with start <= end.

    Returns:
        A list of (start, end) tuples, sorted by start, with no two
        intervals overlapping or touching.
    """
    intervals = list(intervals)
    if not intervals:
        return []

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start < last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]
