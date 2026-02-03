"""Linear variable implementation for structural equation models."""

from torch import Tensor
from typing import Iterable
from steindag.variable.base import Variable


class LinearVariable(Variable):
    """A variable with a linear mean function.

    The mean function is: intercept + sum(coefs[parent] * parent for parent in parents).

    Attributes:
        name: The variable's identifier.
        parent_names: Names of parent variables in the DAG.
        sigma: Standard deviation of the additive noise term.
    """

    def __init__(
        self,
        name: str,
        parent_names: Iterable[str],
        sigma: float,
        coefs: dict[str, float],
        intercept: float,
    ) -> None:
        """Initialize a linear variable.

        Args:
            name: The variable's identifier.
            parent_names: Names of parent variables in the DAG.
            sigma: Standard deviation of the additive noise term.
            coefs: Dictionary mapping parent names to their linear coefficients.
            intercept: The constant term in the linear function.
        """
        super().__init__(name, parent_names, sigma)

        self._coefs = coefs
        self._intercept = intercept

    def f_bar(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the linear mean function.

        Args:
            parents: Dictionary mapping parent names to their tensor values.

        Returns:
            The linear combination: intercept + sum(coefs[parent] * parent).
        """
        f_bar: Tensor | float = self._intercept
        for parent_name, parent in parents.items():
            f_bar = f_bar + self._coefs[parent_name] * parent

        return f_bar  # type: ignore[return-value]
