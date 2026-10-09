from dataclasses import dataclass
from typing import Any


def is_number(value: Any) -> bool:
    """Whether a value is an int or a float, rejecting bools, which subclass `int`."""
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class MiningSettings:
    """
    The parameters a Declare model is mined with.

    - activity_support: The fraction of the traces an activity must occur in to be part of a
        candidate constraint.
    - min_support: The fraction of the traces a candidate must hold in to be kept.
    - max_cardinality: The largest count `n` of Existence, Absence, and Exactly candidates.
    - vacuity: Whether a trace without a constraint's activation satisfies it, both when mining
        and when checking.

    Raises:
        ValueError: If a support lies outside `[0, 1]`, `max_cardinality` is not a positive int,
            or `vacuity` is not a bool.
    """

    activity_support: float
    min_support: float
    max_cardinality: int
    vacuity: bool

    def __post_init__(self) -> None:
        # Store both supports as floats, so that an int threshold saves the same as a float one
        for name in ('activity_support', 'min_support'):
            value = getattr(self, name)
            if not is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} is a number in [0, 1], not {value!r}.')
            object.__setattr__(self, name, float(value))
        if type(self.max_cardinality) is not int or self.max_cardinality < 1:
            raise ValueError(f'max_cardinality is a positive int, not {self.max_cardinality!r}.')
        if type(self.vacuity) is not bool:
            raise ValueError(f'vacuity is a bool, not {self.vacuity!r}.')
