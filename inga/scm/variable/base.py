"""Base classes for structural causal model variables."""

from torch import Tensor
from typing import Iterable


class Variable:
    """A variable in a structural causal model.

    A variable is defined by its name, parent variables, and optional noise
    standard deviation.
    The variable value is computed as f_mean(parents) + sigma * u, where u is a noise term.

    Attributes:
        name: The variable's identifier.
        parent_names: Names of parent variables in the DAG.
        sigma: Standard deviation of the additive noise term.
    """

    def __init__(
        self,
        name: str,
        sigma: float | None = None,
        parent_names: Iterable[str] | None = None,
    ) -> None:
        """Initialize a variable.

        Args:
            name: The variable's identifier.
            sigma: Standard deviation of the additive noise term. Optional when
                only defining DAG structure.
            parent_names: Names of parent variables in the DAG. Defaults to empty list.
        """
        self.name = name
        self.parent_names = list(parent_names) if parent_names is not None else []
        self.sigma = sigma

    def f(
        self, parents: dict[str, Tensor], u: Tensor, f_mean: Tensor | None = None
    ) -> Tensor:
        """Compute the variable value given parents and noise.

        Args:
            parents: Dictionary mapping parent names to their tensor values.
            u: Noise tensor.
            f_mean: Optional precomputed mean function value. If None, it will be computed.

        Raises:
            ValueError: If sigma is not configured.
        """
        if self.sigma is None:
            raise ValueError(
                f"Variable '{self.name}' has no sigma configured. "
                "Set sigma to evaluate structural equations."
            )
        if f_mean is None:
            f_mean = self.f_mean(parents)
        return f_mean + self.sigma * u

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the mean function (expected value given parents).

        Args:
            parents: Dictionary mapping parent names to their tensor values.

        Returns:
            Mean function value.

        Raises:
            NotImplementedError: If no structural function is provided.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' has no structural function configured."
        )
