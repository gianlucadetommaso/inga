"""Structural Equation Model implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from steindag.approx_posterior.laplace import LaplacePosterior
from steindag.sem.causal_bias import CausalBiasMixin
from steindag.sem.html import HTMLMixin
from steindag.sem.plotting import PlottingMixin
from steindag.variable.base import Variable


class SEM(HTMLMixin, PlottingMixin, CausalBiasMixin):
    """A Structural Equation Model (SEM)."""

    def __init__(
        self,
        variables: list[Variable],
        posterior_kwargs: dict | None = None,
    ) -> None:
        self._variables = {variable.name: variable for variable in variables}
        self.posterior = LaplacePosterior(
            variables=self._variables,
            **({} if posterior_kwargs is None else posterior_kwargs),
        )

    def generate(self, num_samples: int) -> dict[str, Tensor]:
        """Generate samples from the SEM by forward sampling."""
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            values[name] = variable.f(parents, torch.randn(num_samples))
        return values

    def posterior_predictive_samples(
        self,
        observed: dict[str, Tensor],
        num_samples: int = 1000,
    ) -> dict[str, Tensor]:
        """Sample all SEM variables from the Laplace posterior predictive.

        Args:
            observed: Mapping of observed variable names to tensors of shape
                ``(batch_size,)``.
            num_samples: Number of posterior samples per batch row.

        Returns:
            Dictionary mapping variable names to tensors of shape
            ``(batch_size, num_samples)``.
        """
        self.posterior.fit(observed)
        latent_samples = self.posterior.sample(num_samples)

        reference = next(iter(observed.values()))
        batch_size = len(reference)
        out: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                parent_name: out[parent_name] for parent_name in variable.parent_names
            }
            if name in observed:
                out[name] = observed[name].unsqueeze(1).expand(batch_size, num_samples)
            else:
                out[name] = variable.f(parents, latent_samples[name])

        return out
