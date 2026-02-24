"""Variable module."""

from inga.scm.variable.base import Variable
from inga.scm.variable.gaussian import GaussianVariable
from inga.scm.variable.categorical import CategoricalVariable
from inga.scm.variable.linear import LinearVariable
from inga.scm.variable.functional import FunctionalVariable

__all__ = [
    "Variable",
    "GaussianVariable",
    "CategoricalVariable",
    "LinearVariable",
    "FunctionalVariable",
]
