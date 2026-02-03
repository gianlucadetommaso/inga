"""Base classes for structural equation model variables."""

from torch import Tensor
from abc import abstractmethod
from typing import Iterable


class Variable:
    """A variable in a structural equation model.

    A variable is defined by its name, parent variables, and noise standard deviation.
    The variable value is computed as f_bar(parents) + sigma * u, where u is a noise term.

    Attributes:
        name: The variable's identifier.
        parent_names: Names of parent variables in the DAG.
        sigma: Standard deviation of the additive noise term.
    """

    def __init__(
        self, name: str, sigma: float, parent_names: Iterable[str] | None = None
    ) -> None:
        """Initialize a variable.

        Args:
            name: The variable's identifier.
            sigma: Standard deviation of the additive noise term.
            parent_names: Names of parent variables in the DAG. Defaults to empty list.
        """
        self.name = name
        self.parent_names = list(parent_names) if parent_names is not None else []
        self.sigma = sigma

    def f(
        self, parents: dict[str, Tensor], u: Tensor, f_bar: Tensor | None = None
    ) -> Tensor:
        """Compute the variable value given parents and noise.

        Args:
            parents: Dictionary mapping parent names to their tensor values.
            u: Noise tensor.
            f_bar: Optional precomputed mean function value. If None, it will be computed.

        Returns:
            The variable value: f_bar(parents) + sigma * u.
        """
        if f_bar is None:
            f_bar = self.f_bar(parents)
        return f_bar + self.sigma * u

    @abstractmethod
    def f_bar(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the mean function (expected value given parents).

        Args:
            parents: Dictionary mapping parent names to their tensor values.

        Returns:
            The mean function value.
        """
        ...
