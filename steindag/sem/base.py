"""Structural Equation Model implementation."""

from torch import Tensor
import torch

from steindag.variable.base import Variable
from steindag.approx_posterior.laplace import LaplacePosterior
from steindag.sem.causal_bias import CausalBiasMixin


class SEM(CausalBiasMixin):
    """A Structural Equation Model (SEM).

    A SEM defines a collection of variables with causal relationships.
    It supports forward sampling, MAP estimation, and approximate posterior inference.

    Attributes:
        _variables: Dictionary mapping variable names to Variable objects.
        posterior: Laplace approximate posterior for inference.
    """

    def __init__(self, variables: list[Variable]) -> None:
        """Initialize the SEM.

        Args:
            variables: List of variables in topological order (parents before children).
        """
        self._variables = {variable.name: variable for variable in variables}
        self.posterior = LaplacePosterior(variables=self._variables)

    def generate(self, num_samples: int) -> dict[str, Tensor]:
        """Generate samples from the SEM by forward sampling.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Dictionary mapping variable names to their sampled tensor values.
        """
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            values[name] = variable.f(parents, torch.randn(num_samples))

        return values
