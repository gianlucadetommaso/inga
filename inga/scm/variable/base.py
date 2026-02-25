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
        # Optional Gaussian scale parameter used by Gaussian-like subclasses.
        # Kept on the base class for static typing convenience across generic
        # SCM utilities that may access `sigma` after runtime validation.
        self.sigma: float | None = None

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

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the deterministic mean component of the structural equation.

        Subclasses with additive-noise structure (e.g. GaussianVariable)
        should implement this method.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' has no mean function configured."
        )

    def f_from_mean(self, f_mean: Tensor, u: Tensor) -> Tensor:
        """Reconstruct value from mean and noise components.

        Subclasses with additive-noise structure can override this for efficient
        recomposition when mean is already available.
        """
        raise NotImplementedError(
            f"Variable '{self.name}' cannot be reconstructed from mean and noise."
        )

    def infer_noise(
        self,
        parents: dict[str, Tensor],
        observed: Tensor,
    ) -> Tensor:
        """Infer exogenous noise from observed value and parent values."""
        raise NotImplementedError(
            f"Variable '{self.name}' does not implement noise inversion."
        )

    def log_pdf(
        self,
        parents: dict[str, Tensor],
        observed: Tensor,
        noise_scale: float = 1.0,
    ) -> Tensor:
        """Log density of observed values under the structural distribution."""
        raise NotImplementedError(
            f"Variable '{self.name}' does not implement log-density evaluation."
        )

    def noise_score(self, u: Tensor) -> Tensor:
        """Score function of exogenous noise: ``∇_u log p(u)``."""
        raise NotImplementedError(
            f"Variable '{self.name}' does not implement a noise score function."
        )
