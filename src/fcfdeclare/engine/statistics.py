from collections.abc import Sequence
from functools import cached_property

import numpy as np


class TraceStatistics:
    """Per-trace statistics of a batch of encoded traces."""

    def __init__(self, traces: Sequence[np.ndarray], n_codes: int) -> None:
        """
        Args:
            traces: The traces of the batch, each an int64 array of codes in `[0, n_codes)`.
            n_codes: How many activity codes the traces use, the reserved code included.
        """
        self.n_traces = len(traces)
        self.n_codes = n_codes
        
        # Traces are first concatenated to a single array of events, 
        # which is more efficient to process than a list of arrays.
        # The leading empty array lets concatenate accept a batch without traces
        self._codes = np.concatenate([np.zeros(0, np.int64), *traces])
        
        # The length of each trace
        self._lengths = np.fromiter(map(len, traces), np.int64, len(traces))
        # The index in the event array of the first event of each trace
        self._starts = np.cumsum(self._lengths) - self._lengths
        # The trace each event belongs to
        self._trace = np.repeat(np.arange(self.n_traces), self._lengths)
        # The position of each event inside its trace, counting from 0
        self._position = np.arange(len(self._codes)) - self._starts[self._trace]

    @cached_property
    def count(self) -> np.ndarray:
        """[T, A] How many times each activity occurs."""
        counts = np.bincount(self._keys, minlength=self.n_traces * self.n_codes)
        return counts.reshape(self.n_traces, self.n_codes)

    @cached_property
    def present(self) -> np.ndarray:
        """[T, A] Whether each activity occurs."""
        return self.count > 0

    @cached_property
    def init(self) -> np.ndarray:
        """[T] The first activity, or -1 for an empty trace."""
        return self._activity_at(self._starts)

    @cached_property
    def end(self) -> np.ndarray:
        """[T] The last activity, or -1 for an empty trace."""
        return self._activity_at(self._starts + self._lengths - 1)

    @cached_property
    def first(self) -> np.ndarray:
        """[T, A] The position of the first occurrence of each activity, or -1 if absent."""
        return self._position_of(self._group_starts)

    @cached_property
    def last(self) -> np.ndarray:
        """[T, A] The position of the last occurrence of each activity, or -1 if absent."""
        return self._position_of(self._group_ends)

    @cached_property
    def bigram(self) -> np.ndarray:
        """[T, A, A] How many times activity `a` is immediately followed by activity `b`."""
        # Two consecutive events form a bigram only if they belong to the same trace
        adjacent = self._trace[1:] == self._trace[:-1]
        keys = (self._trace[:-1] * self.n_codes + self._codes[:-1]) * self.n_codes
        keys = (keys + self._codes[1:])[adjacent]
        counts = np.bincount(keys, minlength=self.n_traces * self.n_codes * self.n_codes)
        return counts.reshape(self.n_traces, self.n_codes, self.n_codes)

    @cached_property
    def alt_response_fail(self) -> np.ndarray:
        """[T, A, A] Whether some `a` has no `b` after it before the next `a` or the trace end."""
        # Each `a` is checked up to the next `a` of its trace, or up to the end of the trace
        trace_ends = self._starts[self._trace] + self._lengths[self._trace]
        following = np.where(self._same_as_next, np.roll(self._order, -1), trace_ends[self._order])
        missing = self._count_between(self._order, following) == 0
        return self._any_per_group(missing)

    @cached_property
    def alt_precedence_fail(self) -> np.ndarray:
        """[T, A, A] Whether some `b` has no `a` before it since the previous `b` or the start."""
        # Each `b` is checked back to the previous `b` of its trace, or to the start of the trace
        before_starts = self._starts[self._trace] - 1
        preceding = np.where(
            self._same_as_previous, np.roll(self._order, 1), before_starts[self._order]
        )
        missing = self._count_between(preceding, self._order) == 0
        # The groups are keyed by `b`, so the axes come out as [T, b, a]
        return self._any_per_group(missing).transpose(0, 2, 1)

    @cached_property
    def _keys(self) -> np.ndarray:
        """[E] The (trace, activity) group of each event, as `trace * A + activity`."""
        return self._trace * self.n_codes + self._codes

    @cached_property
    def _order(self) -> np.ndarray:
        """[E] The events sorted by group, each group in event order thanks to a stable sort."""
        return np.argsort(self._keys, kind='stable')

    @cached_property
    def _same_as_next(self) -> np.ndarray:
        """[E] Whether each sorted event shares its group with the next sorted event."""
        keys = self._keys[self._order]
        same = np.zeros(len(keys), bool)
        same[:-1] = keys[1:] == keys[:-1]
        return same

    @cached_property
    def _same_as_previous(self) -> np.ndarray:
        """[E] Whether each sorted event shares its group with the previous sorted event."""
        # The last sorted event never shares a group with a next one, so the roll brings False
        return np.roll(self._same_as_next, 1)

    @cached_property
    def _group_starts(self) -> np.ndarray:
        """[G] The index in `_order` of the first event of each (trace, activity) group."""
        return np.flatnonzero(~self._same_as_previous)

    @cached_property
    def _group_ends(self) -> np.ndarray:
        """[G] The index in `_order` of the last event of each (trace, activity) group."""
        return np.flatnonzero(~self._same_as_next)

    @cached_property
    def _cumulative(self) -> np.ndarray:
        """[E + 1, A] How many times each activity occurs among the first `i` events."""
        cumulative = np.zeros((len(self._codes) + 1, self.n_codes), np.int64)
        cumulative[np.arange(1, len(self._codes) + 1), self._codes] = 1
        return np.cumsum(cumulative, axis=0, out=cumulative)

    def _activity_at(self, events: np.ndarray) -> np.ndarray:
        """[T] The activity of one event per trace, or -1 for an empty trace."""
        activities = np.full(self.n_traces, -1, np.int64)
        nonempty = self._lengths > 0
        activities[nonempty] = self._codes[events[nonempty]]
        return activities

    def _position_of(self, sorted_events: np.ndarray) -> np.ndarray:
        """[T, A] The position of one sorted event per group, or -1 for an absent activity."""
        positions = np.full(self.n_traces * self.n_codes, -1, np.int64)
        events = self._order[sorted_events]
        positions[self._keys[events]] = self._position[events]
        return positions.reshape(self.n_traces, self.n_codes)

    def _count_between(self, after: np.ndarray, before: np.ndarray) -> np.ndarray:
        """[E, A] How many times each activity occurs strictly between two event indices."""
        return self._cumulative[before] - self._cumulative[after + 1]

    def _any_per_group(self, flags: np.ndarray) -> np.ndarray:
        """[T, A, A] Whether any sorted event of each (trace, activity) group raises a flag."""
        grouped = np.zeros((self.n_traces * self.n_codes, self.n_codes), bool)
        if len(flags):
            starts = self._group_starts
            grouped[self._keys[self._order[starts]]] = np.logical_or.reduceat(flags, starts)
        return grouped.reshape(self.n_traces, self.n_codes, self.n_codes)
