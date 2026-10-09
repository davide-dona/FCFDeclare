from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from fcfdeclare.engine.batching import chunks
from fcfdeclare.engine.encoding import Vocabulary
from fcfdeclare.engine.statistics import TraceStatistics
from fcfdeclare.model.declare_model import DeclareModel
from fcfdeclare.model.templates import Template


@dataclass(frozen=True, slots=True, eq=False)
class Conformance:
    """
    How each trace of a batch scores against a Declare model.

    - satisfied: [B] How many constraints each trace satisfies.
    - total: How many constraints the model has.
    """

    satisfied: np.ndarray
    total: int

    @property
    def share(self) -> np.ndarray:
        """[B] The fraction of the constraints each trace satisfies, or 0 for an empty model."""
        if self.total == 0:
            return np.zeros(len(self.satisfied))
        return self.satisfied / self.total

    @property
    def full(self) -> np.ndarray:
        """
        [B] 1 if a trace satisfies every constraint of a non-empty model, and 0 otherwise.
        A mean over traces is the fraction of them that are conformant.
        """
        return ((self.satisfied == self.total) & (self.total > 0)).astype(np.float64)


@dataclass(frozen=True, slots=True)
class _Group:
    """
    The constraints of a model that share one template, evaluated together.

    - columns: [K] The index of each constraint in the model.
    - a, b, n: [K] Each constraint's activity codes and count, as `Template.evaluate` takes them.
    """

    template: Template
    columns: np.ndarray
    a: np.ndarray
    b: np.ndarray
    n: np.ndarray


class Checker:
    """
    Checks traces against a Declare model, under the vacuity it was mined with.

    Each call evaluates its whole batch with a few numpy operations per template, whose fixed
    cost outweighs the work on a single trace, so pass every trace at once rather than one per
    call. A batch is deduplicated before it is evaluated.
    """

    def __init__(self, model: DeclareModel) -> None:
        self.model = model
        # Code the activities the model names; every other activity takes the reserved code
        self._vocabulary = Vocabulary(
            sorted({name for constraint in model.constraints for name in constraint.activities})
        )

        # Gather the constraints by template, remembering the model column of each
        members: defaultdict[Template, list[tuple[int, int, int, int]]] = defaultdict(list)
        for column, constraint in enumerate(model.constraints):
            codes = [self._vocabulary.codes[name] for name in constraint.activities]
            b = codes[1] if constraint.template.is_binary else 0
            members[constraint.template].append((column, codes[0], b, constraint.n))
        self._groups = tuple(
            _Group(template, *np.array(rows, np.int64).T) for template, rows in members.items()
        )

    @property
    def total(self) -> int:
        """How many constraints the model has."""
        return len(self.model.constraints)

    def check(self, traces: Iterable[Sequence[str]]) -> Conformance:
        """
        Count the constraints each trace satisfies.

        Args:
            traces: The batch, one sequence of activity names per trace. A `str` is a sequence
                of one-character activity names. Activities the model does not name match no
                constraint but still count as events.

        Returns:
            The conformance of each trace, in batch order.
        """
        distinct, inverse = self._distinct(traces)
        # Sum per chunk, so that the full [T, K] matrix is never held at once
        satisfied = np.zeros(len(distinct), np.int64)
        for chunk, holds in self._evaluate(distinct):
            satisfied[chunk] = holds.sum(axis=1)
        return Conformance(satisfied=satisfied[inverse], total=self.total)

    def holds(self, traces: Iterable[Sequence[str]]) -> np.ndarray:
        """
        [B, K] Whether each trace satisfies each constraint, in the order of
        `model.constraints`.

        Args:
            traces: The batch, as `check` takes it.
        """
        distinct, inverse = self._distinct(traces)
        holds = np.zeros((len(distinct), self.total), bool)
        for chunk, chunk_holds in self._evaluate(distinct):
            holds[chunk] = chunk_holds
        return holds[inverse]

    def _distinct(self, traces: Iterable[Sequence[str]]) -> tuple[list[np.ndarray], np.ndarray]:
        """The distinct traces of a batch, encoded, and the index of each trace among them."""
        index: dict[tuple[str, ...], int] = {}
        # A trace seen for the first time takes the next index, a repeated one its earlier index
        inverse = np.fromiter(
            (index.setdefault(tuple(trace), len(index)) for trace in traces), np.int64
        )
        return self._vocabulary.encode(index), inverse

    def _evaluate(self, traces: list[np.ndarray]) -> Iterator[tuple[slice, np.ndarray]]:
        """Yield each chunk of the traces with its [T, K] satisfaction matrix."""
        size = self._vocabulary.size
        for chunk in chunks(traces, size, per_trace=self.total):
            statistics = TraceStatistics(traces[chunk], size)
            holds = np.zeros((statistics.n_traces, self.total), bool)
            # Evaluate each template once over all of its constraints, into their model columns
            for group in self._groups:
                holds[:, group.columns] = group.template.evaluate(
                    statistics, group.a, group.b, group.n, vacuity=self.model.settings.vacuity
                )
            yield chunk, holds
