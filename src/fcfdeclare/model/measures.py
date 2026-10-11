from dataclasses import dataclass

from fcfdeclare.model.settings import is_number


@dataclass(frozen=True, slots=True)
class Measures:
    """
    How strongly the mining log supports a constraint.

    - support: The fraction of the traces in which the constraint's activation occurs and the
        constraint holds.
    - confidence: The fraction of the traces in which the activation occurs that the
        constraint holds in.

    A template without an activation counts every trace as activated, so its confidence equals
    its support. Otherwise the activated traces are a subset of the log, so the support never
    exceeds the confidence.

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
