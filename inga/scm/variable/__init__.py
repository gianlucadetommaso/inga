"""Variable module."""

from inga.scm.variable.base import Variable
from inga.scm.variable.linear import LinearVariable
from inga.scm.variable.functional import FunctionalVariable

__all__ = ["Variable", "LinearVariable", "FunctionalVariable"]
