from dataclasses import dataclass

from fcfdeclare.model.templates import Template

# The position of each template in declaration order, which orders the constraints of a model
_TEMPLATE_RANK = {template: rank for rank, template in enumerate(Template)}


@dataclass(frozen=True, slots=True)
class Constraint:
    """
    A template instantiated on one or two activities.

    - template: The template, or its name.
    - activities: The one or two activity names, in template order: `('a', 'b')` in
        Response(a, b) means that every `a` is followed by a `b`.
    - n: The count of a cardinality template, and 1 for the rest.

    Raises:
        ValueError: If the activities or the count do not fit the template.
    """

    template: Template
    activities: tuple[str, ...]
    n: int = 1

    def __post_init__(self) -> None:
        # Accept a template name and any sequence of activities, stored as a Template and a tuple
        template = Template(self.template)
        activities = tuple(self.activities)
        expected = 2 if template.is_binary else 1
        if len(activities) != expected or not all(isinstance(name, str) for name in activities):
            raise ValueError(f'{template} takes {expected} activity names, not {activities!r}.')
        if template.is_binary and activities[0] == activities[1]:
            raise ValueError(f'{template} takes two distinct activities, not {activities!r}.')
        # Compare exact types, since `bool` is a subclass of `int`
        if type(self.n) is not int or self.n < 1:
            raise ValueError(f'The count of a constraint is a positive int, not {self.n!r}.')
        if self.n != 1 and not template.supports_cardinality:
            raise ValueError(f'{template} takes no count other than 1, not {self.n}.')
        object.__setattr__(self, 'template', template)
        object.__setattr__(self, 'activities', activities)

    def __str__(self) -> str:
        """The constraint as Declare writes it, such as `Response(a, b)` or `Existence(a, 2)`."""
        arguments = list(self.activities)
        if self.template.supports_cardinality:
            arguments.append(str(self.n))
        return f'{self.template}({", ".join(arguments)})'

    @property
    def sort_key(self) -> tuple[int, int, tuple[str, ...]]:
        """The key that orders constraints by template, then by count, then by activity names."""
        return _TEMPLATE_RANK[self.template], self.n, self.activities
