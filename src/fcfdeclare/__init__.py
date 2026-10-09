"""FCFDeclare: fast mining and checking of Declare models over activity sequences."""

from fcfdeclare.checker import Checker, Conformance
from fcfdeclare.miner import mine
from fcfdeclare.model.constraints import Constraint
from fcfdeclare.model.declare_model import DeclareModel
from fcfdeclare.model.settings import MiningSettings
from fcfdeclare.model.templates import Template

__version__ = '0.1.0'

__all__ = [
    'Checker',
    'Conformance',
    'Constraint',
    'DeclareModel',
    'MiningSettings',
    'Template',
    '__version__',
    'mine',
]
