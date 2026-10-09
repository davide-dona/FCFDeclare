from collections.abc import Iterator, Sequence

import numpy as np

# The most array elements the statistics of one chunk may allocate, which bounds peak memory
MEMORY_BUDGET = 32_000_000


def chunks(traces: Sequence[np.ndarray], n_codes: int, per_trace: int = 0) -> Iterator[slice]:
    """
    Split a batch into consecutive slices whose statistics stay within `MEMORY_BUDGET`.

    A trace costs its [A, A] pair statistics, its [length + 1, A] cumulative counts, and whatever
    the caller allocates per trace on top. A slice always holds at least one trace, however
    large, so a single trace above the budget still gets checked.

    Args:
        traces: The encoded traces of the batch.
        n_codes: How many activity codes the traces use, the reserved code included.
        per_trace: How many further elements the caller allocates per trace, such as one row of
            a [T, K] result.
    """
    start, used = 0, 0
    for index, trace in enumerate(traces):
        cost = n_codes * n_codes + (len(trace) + 1) * n_codes + per_trace
        # Close the current slice when the next trace would overflow it
        if index > start and used + cost > MEMORY_BUDGET:
            yield slice(start, index)
            start, used = index, 0
        used += cost
    if start < len(traces):
        yield slice(start, len(traces))
