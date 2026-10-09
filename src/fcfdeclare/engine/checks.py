from collections.abc import Callable

import numpy as np

from fcfdeclare.engine.statistics import TraceStatistics

# A check scores K constraints of one template over a batch of T traces at once: it takes the
# batch statistics and the [K] arrays `a`, `b`, `n` of activity codes and counts, and returns a
# [T, K] bool array. A unary template ignores `b`, and a template without cardinality ignores
# `n`. A template with an activation is checked as if the activation occurs in every trace, so
# its result is meaningful only where it does; `Template.evaluate` handles the other traces.
Check = Callable[[TraceStatistics, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def existence(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` occurs at least `n` times."""
    # Indexing a [T, A] statistic with the [K] array `a` picks one column per constraint: [T, K]
    return s.count[:, a] >= n


def absence(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` occurs fewer than `n` times."""
    return s.count[:, a] < n


def exactly(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` occurs exactly `n` times."""
    return s.count[:, a] == n


def init(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` is the first event of the trace, which an empty trace never satisfies."""
    # Comparing the [T, 1] column with the [K] array `a` broadcasts to [T, K]
    return s.init[:, None] == a


def end(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` is the last event of the trace, which an empty trace never satisfies."""
    return s.end[:, None] == a


def choice(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` or `b` occurs."""
    return s.present[:, a] | s.present[:, b]


def exclusive_choice(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """`a` or `b` occurs, and never both."""
    return s.present[:, a] ^ s.present[:, b]


def responded_existence(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """`b` occurs."""
    return s.present[:, b]


def not_responded_existence(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """`b` does not occur."""
    return ~s.present[:, b]


def response(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """A `b` follows every `a`: the last `b` comes after the last `a`."""
    return s.present[:, b] & (s.last[:, a] < s.last[:, b])


def precedence(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """An `a` precedes every `b`: the first `a` comes before the first `b`."""
    return s.present[:, a] & (s.first[:, a] < s.first[:, b])


def not_response(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """No `b` follows any `a`: `b` is absent or ends before the first `a`."""
    return ~s.present[:, b] | (s.first[:, a] > s.last[:, b])


def not_precedence(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """No `a` precedes any `b`: `a` is absent or starts after the last `b`."""
    return ~s.present[:, a] | (s.first[:, a] > s.last[:, b])


def chain_response(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """A `b` immediately follows every `a`: each `a` starts an `ab` bigram."""
    # Indexing [T, A, A] with `a` and `b` pairs them up, a[k] with b[k], so the result is [T, K]
    return s.bigram[:, a, b] == s.count[:, a]


def chain_precedence(s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """An `a` immediately precedes every `b`: each `b` ends an `ab` bigram."""
    return s.bigram[:, a, b] == s.count[:, b]


def not_chain_response(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """A `b` never immediately follows an `a`."""
    return s.bigram[:, a, b] == 0


def not_chain_precedence(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """An `a` never immediately precedes a `b`."""
    return s.bigram[:, a, b] == 0


def alternate_response(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """A `b` follows every `a` before the next `a` or the end of the trace."""
    return ~s.alt_response_fail[:, a, b]


def alternate_precedence(
    s: TraceStatistics, a: np.ndarray, b: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """An `a` precedes every `b` since the previous `b` or the start."""
    return ~s.alt_precedence_fail[:, a, b]
