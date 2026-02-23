"""Base classes for structural causal model variables."""

from torch import Tensor
from typing import Iterable
from typing import cast


class Variable:
    """A variable in a structural causal model.

    A variable is defined by its name, parent variables, and optional noise
    standard deviation.

    This base class is intentionally agnostic to the noise model. Subclasses
    must implement both :meth:`f_mean` and :meth:`f`.

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

        Subclasses are responsible for specifying the structural/noise model.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' has no noise model configured. "
            "Use a concrete subclass (e.g., GaussianVariable) and/or override `f`."
        )

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


class GaussianVariable(Variable):
    """Variable with additive Gaussian noise.

    The structural equation is:

        value = f_mean(parents) + sigma * u

    where ``u`` is standard normal noise and ``sigma`` is required.
    """

    def __init__(
        self,
        name: str,
        sigma: float,
        parent_names: Iterable[str] | None = None,
    ) -> None:
        if sigma is None:
            raise ValueError(
                f"GaussianVariable '{name}' requires `sigma` and it cannot be None."
            )
        super().__init__(name=name, sigma=sigma, parent_names=parent_names)

    def f(
        self, parents: dict[str, Tensor], u: Tensor, f_mean: Tensor | None = None
    ) -> Tensor:
        """Compute value from mean function and additive Gaussian noise."""
        if f_mean is None:
            f_mean = self.f_mean(parents)
        sigma = cast(float, self.sigma)
        return f_mean + sigma * u
