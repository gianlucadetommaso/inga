"""Gaussian variable implementation for structural causal models."""

import torch
from torch import Tensor
from typing import Iterable, cast

from inga.scm.variable.base import Variable


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
        super().__init__(name=name, parent_names=parent_names)
        self.sigma = sigma

    def f(self, parents: dict[str, Tensor], u: Tensor) -> Tensor:
        """Compute value from mean function and additive Gaussian noise."""
        f_mean = self.f_mean(parents)
        return self.f_from_mean(f_mean=f_mean, u=u)

    def f_from_mean(self, f_mean: Tensor, u: Tensor) -> Tensor:
        """Combine precomputed mean and noise into the structural value.

        This is useful in routines that already computed ``f_mean`` and want to
        avoid recomputing it when forming ``f_mean + sigma * u``.
        """
        sigma = cast(float, self.sigma)
        return f_mean + sigma * u

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the mean function for Gaussian structural equations."""
        raise NotImplementedError(
            f"GaussianVariable '{self.name}' has no structural mean function configured."
        )

    def sample_noise(
        self,
        num_samples: int,
        parents: dict[str, Tensor],
    ) -> Tensor:
        """Sample standard Gaussian noise."""
        return torch.randn(num_samples)

    def infer_noise(
        self,
        parents: dict[str, Tensor],
        observed: Tensor,
    ) -> Tensor:
        """Infer additive Gaussian noise from observed values."""
        sigma = cast(float, self.sigma)
        return (observed - self.f_mean(parents)) / sigma
