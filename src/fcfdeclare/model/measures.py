from dataclasses import dataclass

from fcfdeclare.model.settings import is_number


@dataclass(frozen=True, slots=True)
class Measures:
    """
    How strongly the mining log supports a constraint.

    - support: The fraction of the traces in which the constraint is non-vacuously satisfied:
        its activation occurs and it holds.
    - confidence: The ratio between the support of the constraint and the support of its
        activation, the fraction of the traces in which the activation occurs.

    A template without an activation has an activation support of 1, so its confidence equals
    its support. Otherwise the activation support is at most 1, so the support never exceeds
    the confidence.

    Raises:
        ValueError: If a measure lies outside `[0, 1]`, or the support exceeds the confidence.
    """

    support: float
    confidence: float

    def __post_init__(self) -> None:
        # Store both measures as floats, so that an int measure saves the same as a float one
        for name in ('support', 'confidence'):
            value = getattr(self, name)
            if not is_number(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} is a number in [0, 1], not {value!r}.')
            object.__setattr__(self, name, float(value))
        if self.support > self.confidence:
            raise ValueError(
                f'The support {self.support} exceeds the confidence {self.confidence}.'
            )
