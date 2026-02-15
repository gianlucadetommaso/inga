"""Structural Equation Model implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from steindag.approx_posterior.laplace import LaplacePosterior
from steindag.sem.causal_bias import CausalBiasMixin
from steindag.sem.plotting import PlottingMixin
from steindag.variable.base import Variable


class SEM(PlottingMixin, CausalBiasMixin):
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
