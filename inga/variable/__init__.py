"""Variable module."""

from inga.variable.base import Variable
from inga.variable.linear import LinearVariable
from inga.variable.functional import FunctionalVariable

__all__ = ["Variable", "LinearVariable", "FunctionalVariable"]
