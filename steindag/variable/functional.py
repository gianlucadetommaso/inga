"""Functional variable implementation for structural equation models."""

from __future__ import annotations

from typing import Callable, Iterable

from torch import Tensor

from steindag.variable.base import Variable


class FunctionalVariable(Variable):
    """A variable with an arbitrary mean function.

    The mean function is provided as a callable mapping parent values to a tensor.
    This supports nonlinear transformations composed from known functions like
    sin, cos, exp, and tanh.
    """

    def __init__(
        self,
        name: str,
        sigma: float,
        f_mean: Callable[[dict[str, Tensor]], Tensor],
        parent_names: Iterable[str] | None = None,
        coefs: dict[str, float] | None = None,
        intercept: float | None = None,
        transforms: list[str] | None = None,
    ) -> None:
        """Initialize a functional variable.

        Args:
            name: The variable's identifier.
            sigma: Standard deviation of the additive noise term.
            f_mean: Callable computing the mean function given parent values.
            parent_names: Names of parent variables in the DAG. Defaults to empty list.
            coefs: Optional linear coefficients used to build the mean function.
            intercept: Optional intercept used to build the mean function.
            transforms: Optional list of transform names applied to the mean function.
        """
        super().__init__(name, sigma, parent_names)
        self._f_mean = f_mean
        self._coefs = coefs
        self._intercept = intercept
        self._transforms = transforms

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the mean function using the provided callable."""
        return self._f_mean(parents)
