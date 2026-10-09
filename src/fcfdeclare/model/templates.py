from enum import StrEnum
from typing import Self

import numpy as np

from fcfdeclare.engine import checks
from fcfdeclare.engine.checks import Check
from fcfdeclare.engine.statistics import TraceStatistics


class Template(StrEnum):
    """
    A Declare template. Its value is the name a saved model stores.

    Each member is declared as `name, check, is_binary, supports_cardinality, activation`.

    - check: The template's vectorized check, which assumes the activation occurs.
    - is_binary: Whether the template takes two activities rather than one.
    - supports_cardinality: Whether the template takes a count `n` other than 1.
    - activation: The index of the activity that activates the template, `0` for `a` and `1`
        for `b`, or None for a template without an activation.
    """

    check: Check
    is_binary: bool
    supports_cardinality: bool
    activation: int | None

    def __new__(
        cls,
        name: str,
        check: Check,
        is_binary: bool,
        supports_cardinality: bool,
        activation: int | None,
    ) -> Self:
        member = str.__new__(cls, name)
        member._value_ = name
        member.check = check
        member.is_binary = is_binary
        member.supports_cardinality = supports_cardinality
        member.activation = activation
        return member

    EXISTENCE = 'Existence', checks.existence, False, True, None
    ABSENCE = 'Absence', checks.absence, False, True, None
    EXACTLY = 'Exactly', checks.exactly, False, True, None
    INIT = 'Init', checks.init, False, False, None
    END = 'End', checks.end, False, False, None
    CHOICE = 'Choice', checks.choice, True, False, None
    EXCLUSIVE_CHOICE = 'Exclusive Choice', checks.exclusive_choice, True, False, None
    RESPONDED_EXISTENCE = 'Responded Existence', checks.responded_existence, True, False, 0
    NOT_RESPONDED_EXISTENCE = (
        'Not Responded Existence',
        checks.not_responded_existence,
        True,
        False,
        0,
    )
    RESPONSE = 'Response', checks.response, True, False, 0
    PRECEDENCE = 'Precedence', checks.precedence, True, False, 1
    NOT_RESPONSE = 'Not Response', checks.not_response, True, False, 0
    NOT_PRECEDENCE = 'Not Precedence', checks.not_precedence, True, False, 1
    CHAIN_RESPONSE = 'Chain Response', checks.chain_response, True, False, 0
    CHAIN_PRECEDENCE = 'Chain Precedence', checks.chain_precedence, True, False, 1
    NOT_CHAIN_RESPONSE = 'Not Chain Response', checks.not_chain_response, True, False, 0
    NOT_CHAIN_PRECEDENCE = 'Not Chain Precedence', checks.not_chain_precedence, True, False, 1
    ALTERNATE_RESPONSE = 'Alternate Response', checks.alternate_response, True, False, 0
    ALTERNATE_PRECEDENCE = 'Alternate Precedence', checks.alternate_precedence, True, False, 1

    def evaluate(
        self,
        statistics: TraceStatistics,
        a: np.ndarray,
        b: np.ndarray,
        n: np.ndarray,
        *,
        vacuity: bool,
    ) -> np.ndarray:
        """
        [T, K] Whether each trace of a batch satisfies each of K constraints of this template.

        Args:
            statistics: The statistics of the batch.
            a: [K] The code of each constraint's first activity.
            b: [K] The code of each constraint's second activity, ignored by a unary template.
            n: [K] The count of each constraint, ignored by a template without cardinality.
            vacuity: Whether a trace without a constraint's activation satisfies it.

        Example:
            Two traces over the codes 0, 1, 2: `0 1` and `2`, checked against the K = 2
            constraints Response(0, 1) and Response(1, 0). Each row is a trace and each column
            a constraint.

            >>> statistics = TraceStatistics([np.array([0, 1]), np.array([2])], n_codes=3)
            >>> a, b, n = np.array([0, 1]), np.array([1, 0]), np.array([1, 1])
            >>> Template.RESPONSE.evaluate(statistics, a, b, n, vacuity=False)
            array([[ True, False],
                   [False, False]])

            The second trace has no activation for either constraint, so with vacuity it
            satisfies both.

            >>> Template.RESPONSE.evaluate(statistics, a, b, n, vacuity=True)
            array([[ True, False],
                   [ True,  True]])
        """
        holds = self.check(statistics, a, b, n)
        if self.activation is None:
            return holds
        activated = statistics.present[:, (a, b)[self.activation]]
        # A trace without the activation satisfies the constraint only vacuously
        return holds | ~activated if vacuity else holds & activated
