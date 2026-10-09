import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

from fcfdeclare.model import serialization
from fcfdeclare.model.constraints import Constraint
from fcfdeclare.model.settings import MiningSettings, is_number


@dataclass(frozen=True, slots=True)
class DeclareModel:
    """
    A Declare model, with the settings it was mined under.

    - settings: The mining settings, whose `vacuity` checking reuses.
    - constraints: Each constraint, to its support: the fraction of the mining log's traces that
        satisfy it. The model stores them read-only, ordered by template in `Template` order,
        then by count, then by activity names, which fixes both the saved file and the columns
        of `Checker.holds`.

    Raises:
        ValueError: If a key is not a `Constraint` or a support lies outside `[0, 1]`.
    """

    settings: MiningSettings
    constraints: Mapping[Constraint, float]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, MiningSettings):
            raise ValueError(f'The settings are MiningSettings, not {self.settings!r}.')
        for constraint in self.constraints:
            if not isinstance(constraint, Constraint):
                raise ValueError(f'The constraints are keyed by Constraint, not {constraint!r}.')

        # Store the constraints in canonical order, so that equal models serialize identically
        constraints: dict[Constraint, float] = {}
        for constraint, support in sorted(
            self.constraints.items(), key=lambda item: item[0].sort_key
        ):
            if not is_number(support) or not 0.0 <= support <= 1.0:
                raise ValueError(f'The support of {constraint} is {support!r}, not in [0, 1].')
            constraints[constraint] = float(support)
        object.__setattr__(self, 'constraints', MappingProxyType(constraints))

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the model as JSON, which equal models always serialize to the same bytes."""
        text = serialization.dumps(self.settings, self.constraints)
        Path(path).write_text(text, encoding='utf-8')

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """
        Load a model saved with `save`.

        Raises:
            ValueError: If the file is not a well-formed model.
        """
        try:
            text = Path(path).read_text(encoding='utf-8')
            settings, constraints = serialization.loads(text)
            return cls(settings, constraints)
        except ValueError as error:
            raise ValueError(f'{path} is not a valid Declare model: {error}') from error
