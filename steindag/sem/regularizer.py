"""Regularizers for SEM causal bias."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

from steindag.sem.base import SEM


@dataclass(frozen=True)
class CausalRegularizer:
    """Wrapper to compute a causal-bias regularization term.

    This adapts models that accept a 2D tensor of observed variables into the
    dictionary-based interface expected by the SEM causal bias routines.
    """

    sem: SEM
    model: Callable[[Tensor], Tensor]
    observed_names: Sequence[str]
    treatment_name: str
    outcome_name: str

    def ravel_observed(self, observed: dict[str, Tensor]) -> Tensor:
        """Stack observed variables into a 2D tensor (n_obs, n_features)."""
        return torch.stack([observed[name] for name in self.observed_names], dim=1)

    def unravel_observed(self, observed_tensor: Tensor) -> dict[str, Tensor]:
        """Convert a 2D tensor back into an observed dict."""
        return {
            name: observed_tensor.select(dim=1, index=i)
            for i, name in enumerate(self.observed_names)
        }

    def conditional_mean(self, observed: dict[str, Tensor]) -> Tensor:
        """Compute conditional mean using the wrapped model."""
        inputs = self.ravel_observed(observed)
        outputs = self.model(inputs)
        if outputs.ndim == 2 and outputs.shape[1] == 1:
            return outputs.squeeze(1)
        return outputs

    def causal_bias(
        self,
        observed: dict[str, Tensor],
        num_samples: int = 1000,
        enable_grad: bool = True,
    ) -> Tensor:
        """Compute causal bias using the wrapped model."""
        return self.sem.causal_bias(
            observed,
            treatment_name=self.treatment_name,
            outcome_name=self.outcome_name,
            num_samples=num_samples,
            conditional_mean_fn=self.conditional_mean,
            enable_grad=enable_grad,
        )

    def regularization_term(
        self,
        observed: dict[str, Tensor],
        num_samples: int = 1000,
        reduction: str = "mean",
        enable_grad: bool = True,
    ) -> Tensor:
        """Compute a regularization term from the causal bias.

        Args:
            observed: Observed variables used for the bias computation.
            num_samples: Number of posterior samples for causal bias.
            reduction: "mean", "sum", or "none".
        """
        bias = self.causal_bias(
            observed, num_samples=num_samples, enable_grad=enable_grad
        )
        penalty = bias.abs()

        if reduction == "mean":
            return penalty.mean()
        if reduction == "sum":
            return penalty.sum()
        if reduction == "none":
            return penalty

        raise ValueError("reduction must be one of 'mean', 'sum', or 'none'.")
