"""Structural Causal Model implementation."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from inga.approx_posterior.laplace import LaplacePosterior
from inga.scm.causal_bias import CausalBiasMixin
from inga.scm.dataset_core import (
    generate_dataset_from_scm as _generate_dataset_from_scm,
)
from inga.scm.html import HTMLMixin
from inga.scm.plotting import PlottingMixin
from inga.scm.variable.base import Variable

if TYPE_CHECKING:
    from inga.scm.dataset_core import CausalQueryConfig, SCMDataset


class SCM(HTMLMixin, PlottingMixin, CausalBiasMixin):
    """A Structural Causal Model (SCM)."""

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
        """Generate samples from the SCM by forward sampling."""
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            f_mean = variable.f_mean(parents)
            noise = variable.sample_noise(num_samples, parents, f_mean=f_mean)
            values[name] = variable.f(parents, noise, f_mean=f_mean)
        return values

    def posterior_predictive_samples(
        self,
        observed: dict[str, Tensor],
        num_samples: int = 1000,
    ) -> dict[str, Tensor]:
        """Sample all SCM variables from the Laplace posterior predictive.

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

    def generate_dataset(
        self,
        num_samples: int = 1000,
        num_queries: int | None = None,
        min_observed: int = 1,
        seed: int | None = None,
        treatment_name: str | None = None,
        queries: list[CausalQueryConfig] | None = None,
    ) -> SCMDataset:
        """Generate a causal dataset directly from this SCM.

        This is a convenience wrapper around
        :func:`inga.scm.dataset.generate_dataset_from_scm`.
        """
        return _generate_dataset_from_scm(
            scm=self,
            num_samples=num_samples,
            num_queries=num_queries,
            min_observed=min_observed,
            seed=seed,
            treatment_name=treatment_name,
            queries=queries,
        )
