import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any

from fcfdeclare.model.constraints import Constraint
from fcfdeclare.model.settings import MiningSettings

# The exact keys of a saved model, of its settings, and of each of its constraints
_MODEL_KEYS = {'settings', 'constraints'}
_SETTINGS_KEYS = {field.name for field in fields(MiningSettings)}
_CONSTRAINT_KEYS = {'template', 'activities', 'n', 'support'}


def dumps(settings: MiningSettings, constraints: Mapping[Constraint, float]) -> str:
    """Write a model as JSON, its constraints in the order of `constraints`."""
    payload = {
        'settings': asdict(settings),
        'constraints': [
            {
                'template': str(constraint.template),
                'activities': list(constraint.activities),
                'n': constraint.n,
                'support': support,
            }
            for constraint, support in constraints.items()
        ],
    }
    return json.dumps(payload, indent=4, ensure_ascii=False) + '\n'


def loads(text: str) -> tuple[MiningSettings, dict[Constraint, Any]]:
    """
    Read a model written by `dumps`.

    Returns:
        The settings and each constraint to its support, which is returned as read for
        `DeclareModel` to validate.

    Raises:
        ValueError: If the text is not a saved model.
    """
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != _MODEL_KEYS:
        raise ValueError(f'a model has exactly the keys {sorted(_MODEL_KEYS)}.')

    settings = payload['settings']
    if not isinstance(settings, dict) or set(settings) != _SETTINGS_KEYS:
        raise ValueError(f'the settings have exactly the keys {sorted(_SETTINGS_KEYS)}.')

    entries = payload['constraints']
    if not isinstance(entries, list):
        raise ValueError('the constraints are a list.')
    constraints = dict(_constraint(entry) for entry in entries)
    # A repeated constraint collapses into a single key, so the dict comes out shorter
    if len(constraints) != len(entries):
        raise ValueError('a constraint is repeated.')

    return MiningSettings(**settings), constraints


def _constraint(entry: Any) -> tuple[Constraint, Any]:
    """Parse one saved constraint and its support, which `Constraint` validates in turn."""
    if not isinstance(entry, dict) or set(entry) != _CONSTRAINT_KEYS:
        raise ValueError(f'{entry!r} does not have exactly the keys {sorted(_CONSTRAINT_KEYS)}.')
    if not isinstance(entry['activities'], list):
        raise ValueError(f'the activities of {entry!r} are not a list.')
    constraint = Constraint(entry['template'], tuple(entry['activities']), entry['n'])
    return constraint, entry['support']
