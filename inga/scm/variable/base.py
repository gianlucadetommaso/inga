"""Base classes for structural causal model variables."""

from torch import Tensor
from typing import Iterable


class Variable:
    """A variable in a structural causal model.

    A variable is defined by its name and parent variables.

    This base class is intentionally agnostic to the noise model. Subclasses
    must implement :meth:`f` and :meth:`sample_noise`.

    Attributes:
        name: The variable's identifier.
        parent_names: Names of parent variables in the DAG.
    """

    def __init__(
        self,
        name: str,
        parent_names: Iterable[str] | None = None,
    ) -> None:
        """Initialize a variable.

        Args:
            name: The variable's identifier.
            parent_names: Names of parent variables in the DAG. Defaults to empty list.
        """
        self.name = name
        self.parent_names = list(parent_names) if parent_names is not None else []

    def f(
        self,
        parents: dict[str, Tensor],
        u: Tensor,
    ) -> Tensor:
        """Compute the variable value given parents and exogenous noise.

        This base class intentionally does not define any structural equation.
        Subclasses are responsible for specifying the full forward model.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' has no noise model configured. "
            "Use a concrete subclass (e.g., GaussianVariable) and/or override `f`."
        )

    def sample_noise(
        self,
        num_samples: int,
        parents: dict[str, Tensor],
    ) -> Tensor:
        """Sample exogenous noise for this variable.

        Subclasses should implement the distribution used for their structural
        equations.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' has no noise sampler configured."
        )
