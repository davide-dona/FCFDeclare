from dataclasses import dataclass
from typing import Any


def is_number(value: Any) -> bool:
    """Whether a value is an int or a float, rejecting bools, which subclass `int`."""
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class MiningSettings:
    """
    The parameters a Declare model is mined with. Each threshold filters on its own measure,
    so changing one never changes what another means.

    - min_support: The least support a candidate needs to be kept: the fraction of the traces
        in which it is non-vacuously satisfied.
    - min_confidence: The least confidence a candidate needs to be kept: the ratio between its
        support and the support of its activation.
    - max_cardinality: The largest count `n` of Existence, Absence, and Exactly candidates.

    Raises:
        ValueError: If a threshold lies outside `[0, 1]` or `max_cardinality` is not a
            positive int.
    """

    min_support: float
    min_confidence: float
    max_cardinality: int

    def __post_init__(self) -> None:
        # Store both thresholds as floats, so that an int threshold saves the same as a float one
        for name in ('min_support', 'min_confidence'):
            value = getattr(self, name)
            if not is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} is a number in [0, 1], not {value!r}.')
            object.__setattr__(self, name, float(value))
        if type(self.max_cardinality) is not int or self.max_cardinality < 1:
            raise ValueError(f'max_cardinality is a positive int, not {self.max_cardinality!r}.')
