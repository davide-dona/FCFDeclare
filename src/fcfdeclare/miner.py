from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from fcfdeclare.engine.batching import chunks
from fcfdeclare.engine.encoding import Vocabulary
from fcfdeclare.engine.statistics import TraceStatistics
from fcfdeclare.model.constraints import Constraint
from fcfdeclare.model.declare_model import DeclareModel
from fcfdeclare.model.settings import MiningSettings
from fcfdeclare.model.templates import Template


def mine(
    traces: Iterable[Sequence[str]],
    *,
    activity_support: float,
    min_support: float,
    max_cardinality: int,
    vacuity: bool,
) -> DeclareModel:
    """
    Mine a Declare model from a log.

    Every activity that occurs in at least `activity_support` of the traces is frequent. The
    candidates are each template on every frequent activity, or on every ordered pair of distinct
    frequent activities, with counts from 1 to `max_cardinality` for a cardinality template.
    A candidate is kept if at least `min_support` of the traces satisfy it.

    Args:
        traces: The log, one sequence of activity names per case. A `str` is a sequence of
            one-character activity names.
        activity_support: See `MiningSettings`.
        min_support: See `MiningSettings`.
        max_cardinality: See `MiningSettings`.
        vacuity: See `MiningSettings`.

    Returns:
        The mined model, each constraint with its support: the fraction of the traces that
        satisfy it.

    Raises:
        ValueError: If the log is empty or a setting is invalid.
    """
    settings = MiningSettings(
        activity_support=activity_support,
        min_support=min_support,
        max_cardinality=max_cardinality,
        vacuity=vacuity,
    )

    # Count the distinct variants, so that each is checked once and weighted by its frequency
    variants = Counter(tuple(trace) for trace in traces)
    n_traces = sum(variants.values())
    if n_traces == 0:
        raise ValueError('Cannot mine a Declare model from an empty log.')

    # Keep the activities that occur in at least `activity_support` of the traces
    traces_with: Counter[str] = Counter()
    for variant, weight in variants.items():
        for activity in set(variant):
            traces_with[activity] += weight
    frequent = sorted(
        activity
        for activity, count in traces_with.items()
        if count / n_traces >= settings.activity_support
    )

    # Code the frequent activities in name order, so that the candidates come out sorted. The
    # rare ones share the reserved code, since they only matter as events
    vocabulary = Vocabulary(frequent)
    candidates = {
        template: _candidates(template, len(frequent), settings.max_cardinality)
        for template in Template
    }

    # Count the traces that satisfy each candidate, one chunk of variants at a time
    encoded = vocabulary.encode(variants)
    weights = np.fromiter(variants.values(), np.int64, len(variants))
    satisfying = {
        template: np.zeros(len(a), np.int64) for template, (a, _, _) in candidates.items()
    }
    # The [T, K] result of the template with the most candidates is the largest per-trace array
    largest = max(len(a) for a, _, _ in candidates.values())
    for chunk in chunks(encoded, vocabulary.size, per_trace=largest):
        statistics = TraceStatistics(encoded[chunk], vocabulary.size)
        for template, (a, b, n) in candidates.items():
            holds = template.evaluate(statistics, a, b, n, vacuity=settings.vacuity)
            satisfying[template] += weights[chunk] @ holds

    # Keep the candidates that at least `min_support` of the traces satisfy. Integer counts
    # divided once give supports exactly equal to a direct count over the log
    constraints: dict[Constraint, float] = {}
    for template, (a, b, n) in candidates.items():
        supports = satisfying[template] / n_traces
        for k in np.flatnonzero(supports >= settings.min_support):
            codes = (a[k], b[k]) if template.is_binary else (a[k],)
            names = tuple(vocabulary.names[code] for code in codes)
            constraints[Constraint(template, names, int(n[k]))] = float(supports[k])
    return DeclareModel(settings=settings, constraints=constraints)


def _candidates(
    template: Template, n_activities: int, max_cardinality: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The candidates of one template, as the [K] arrays of their activity codes `a` and `b` and
    their counts `n`, ordered by count and then by activities.
    """
    # A binary template takes every ordered pair of distinct activities, with no co-occurrence
    # requirement
    if template.is_binary:
        a, b = np.nonzero(~np.eye(n_activities, dtype=bool))
        return a, b, np.ones(len(a), np.int64)
    # A unary template takes every activity, once per count if it supports cardinality
    cardinalities = max_cardinality if template.supports_cardinality else 1
    a = np.tile(np.arange(n_activities), cardinalities)
    n = np.repeat(np.arange(1, cardinalities + 1), n_activities)
    return a, np.zeros_like(a), n
