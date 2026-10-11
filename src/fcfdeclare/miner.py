from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from fcfdeclare.engine.batching import chunks
from fcfdeclare.engine.encoding import Vocabulary
from fcfdeclare.engine.statistics import TraceStatistics
from fcfdeclare.model.constraints import Constraint
from fcfdeclare.model.declare_model import DeclareModel
from fcfdeclare.model.measures import Measures
from fcfdeclare.model.settings import MiningSettings
from fcfdeclare.model.templates import Template


def mine(
    traces: Iterable[Sequence[str]],
    *,
    min_support: float,
    min_confidence: float,
    max_cardinality: int,
) -> DeclareModel:
    """
    Mine a Declare model from a log.

    The candidates are each template on every activity of the log, or on every ordered pair of
    distinct activities, with counts from 1 to `max_cardinality` for a cardinality template.
    A candidate is kept if its support reaches `min_support` and its confidence reaches
    `min_confidence`.

    Args:
        traces: The log, one sequence of activity names per case. A `str` is a sequence of
            one-character activity names.
        min_support: See `MiningSettings`.
        min_confidence: See `MiningSettings`.
        max_cardinality: See `MiningSettings`.

    Returns:
        The mined model, each constraint with its support and confidence.

    Raises:
        ValueError: If the log is empty or a setting is invalid.
    """
    settings = MiningSettings(
        min_support=min_support,
        min_confidence=min_confidence,
        max_cardinality=max_cardinality,
    )

    # Count the distinct variants, so that each is checked once and weighted by its frequency
    variants = Counter(tuple(trace) for trace in traces)
    n_traces = sum(variants.values())
    if n_traces == 0:
        raise ValueError('Cannot mine a Declare model from an empty log.')

    # Count the traces each activity occurs in, which activate the constraints it activates
    traces_with: Counter[str] = Counter()
    for variant, weight in variants.items():
        for activity in set(variant):
            traces_with[activity] += weight

    # Code the activities in name order, so that the candidates come out sorted
    vocabulary = Vocabulary(sorted(traces_with))
    activated = np.fromiter(
        (traces_with[name] for name in vocabulary.names), np.int64, len(vocabulary.names)
    )
    candidates = {
        template: _candidates(template, len(vocabulary.names), settings.max_cardinality)
        for template in Template
    }

    # Count the traces that witness each candidate, one chunk of variants at a time
    encoded = vocabulary.encode(variants)
    weights = np.fromiter(variants.values(), np.int64, len(variants))
    witnesses = {template: np.zeros(len(a), np.int64) for template, (a, _, _) in candidates.items()}
    # The [T, K] result of the template with the most candidates is the largest per-trace array
    largest = max(len(a) for a, _, _ in candidates.values())
    for chunk in chunks(encoded, vocabulary.size, per_trace=largest):
        statistics = TraceStatistics(encoded[chunk], vocabulary.size)
        for template, (a, b, n) in candidates.items():
            witnesses[template] += weights[chunk] @ template.witness(statistics, a, b, n)

    # Keep the candidates that reach both thresholds. Integer counts divided once give measures
    # exactly equal to a direct count over the log. Every activity occurs in some trace, so no
    # candidate has zero activated traces
    constraints: dict[Constraint, Measures] = {}
    for template, (a, b, n) in candidates.items():
        activation = template.activation_of(a, b)
        activations = n_traces if activation is None else activated[activation]
        supports = witnesses[template] / n_traces
        confidences = witnesses[template] / activations
        kept = (supports >= settings.min_support) & (confidences >= settings.min_confidence)
        for k in np.flatnonzero(kept):
            codes = (a[k], b[k]) if template.is_binary else (a[k],)
            names = tuple(vocabulary.names[code] for code in codes)
            constraints[Constraint(template, names, int(n[k]))] = Measures(
                float(supports[k]), float(confidences[k])
            )
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
