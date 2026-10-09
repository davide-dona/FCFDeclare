from collections.abc import Iterable, Sequence

import numpy as np


class Vocabulary:
    """
    A mapping from activity names to integer codes, used by vectorized checks.

    The known names take the codes `0` to `len(names) - 1` in the given order.
    Every other activity takes the reserved code `len(names)`. No constraint are
    matched by it, but still counts as an event.

    Example:
        A `str` trace is a sequence of one-character activities, and `x` and `z` share the
        reserved code 3.

        >>> vocabulary = Vocabulary(['a', 'b', 'c'])
        >>> vocabulary.codes
        {'a': 0, 'b': 1, 'c': 2}
        >>> vocabulary.encode(['acax', ['b', 'z']])
        [array([0, 2, 0, 3]), array([1, 3])]
    """

    def __init__(self, names: Sequence[str]) -> None:
        """
        Args:
            names: The distinct activity names, in code order.
        """
        self.names = tuple(names)
        # A dict mapping each name to its code, for fast encoding.
        # The reserved code is not included.
        self.codes = {name: code for code, name in enumerate(self.names)}
        # The code shared by every activity outside `names`
        self.unknown = len(self.names)

    @property
    def size(self) -> int:
        """How many codes a trace can use, the reserved code included."""
        return len(self.names) + 1

    def encode(self, traces: Iterable[Sequence[str]]) -> list[np.ndarray]:
        """Encode each trace as an int64 array of activity codes."""
        return [
            np.fromiter(
                (self.codes.get(activity, self.unknown) for activity in trace),
                np.int64,
                len(trace),
            )
            for trace in traces
        ]
