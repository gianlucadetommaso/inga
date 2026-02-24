"""Categorical variable implementation for structural causal models."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor

from inga.scm.variable.base import Variable


class CategoricalVariable(Variable):
    """A variable with a straight-through categorical structural equation.

    Given ``V_circ = f_logits(parents) + U`` (with logits in the last dimension),
    this variable computes:

    - ``V_bar = one_hot(argmax(V_circ))``
    - ``V_tilde = softmax(V_circ)``
    - ``V = V_tilde + stop_gradient(V_bar - V_tilde)``

    The forward pass is exactly one-hot (``V_bar``), while the backward pass uses
    the softmax path (``V_tilde``).
    """

    def __init__(
        self,
        name: str,
        f_logits: Callable[[dict[str, Tensor]], Tensor],
        parent_names: Iterable[str] | None = None,
        temperature: float = 0.1,
    ) -> None:
        if temperature <= 0:
            raise ValueError("`temperature` must be positive.")
        super().__init__(name=name, parent_names=parent_names)
        self._f_logits = f_logits
        self._temperature = temperature

    def f_logits(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute logits from parent values."""
        return self._f_logits(parents)

    def f(
        self,
        parents: dict[str, Tensor],
        u: Tensor,
    ) -> Tensor:
        """Compute straight-through one-hot values from logits plus noise."""
        structural = self.f_logits(parents)

        v_circ = self._combine_logits_and_noise(structural, u)
        v_tilde = torch.softmax(v_circ / self._temperature, dim=-1)
        indices = torch.argmax(v_circ, dim=-1)
        v_bar = torch.nn.functional.one_hot(
            indices, num_classes=v_circ.shape[-1]
        ).to(v_tilde.dtype)
        return v_tilde + (v_bar - v_tilde).detach()

    def sample_noise(
        self,
        num_samples: int,
        parents: dict[str, Tensor],
    ) -> Tensor:
        """Sample Gumbel noise for categorical structural equations."""
        logits = self.f_logits(parents)
        if logits.ndim == 1:
            shape = (num_samples, logits.shape[0])
        elif logits.ndim >= 2:
            shape = tuple(logits.shape)
        else:
            raise ValueError(
                "Categorical logits must have at least one dimension (categories)."
            )

        uniform = torch.rand(shape, device=logits.device, dtype=logits.dtype)
        eps = torch.finfo(logits.dtype).eps
        uniform = uniform.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(uniform))

    @staticmethod
    def _combine_logits_and_noise(logits: Tensor, noise: Tensor) -> Tensor:
        """Broadcast/add noise to logits, supporting class-dimension expansion."""
        if logits.ndim == 1 and noise.ndim >= 2 and noise.shape[-1] == logits.shape[0]:
            logits = logits.unsqueeze(0).expand(noise.shape[0], -1)

        if logits.ndim == 1 and noise.ndim == 1 and logits.shape[0] != noise.shape[0]:
            logits = logits.unsqueeze(0).expand(noise.shape[0], -1)

        if logits.ndim == noise.ndim + 1 and logits.shape[:-1] == noise.shape:
            noise = noise.unsqueeze(-1).expand_as(logits)

        if logits.shape != noise.shape:
            try:
                noise = noise.expand_as(logits)
            except RuntimeError as exc:
                raise ValueError(
                    "Noise tensor shape is incompatible with categorical logits. "
                    f"Got logits shape {tuple(logits.shape)} and noise shape {tuple(noise.shape)}."
                ) from exc

        return logits + noise
