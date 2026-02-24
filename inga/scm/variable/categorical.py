"""Categorical variable implementation for structural causal models."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import Tensor

from inga.scm.variable.base import Variable


class CategoricalVariable(Variable):
    """A variable with a straight-through categorical structural equation.

    Given ``V_circ = f_mean(parents) + U`` (with logits in the last dimension),
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
        f_mean: Callable[[dict[str, Tensor]], Tensor],
        parent_names: Iterable[str] | None = None,
        sigma: float = 1.0,
    ) -> None:
        super().__init__(name=name, sigma=sigma, parent_names=parent_names)
        self._f_mean = f_mean

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute logits from parent values."""
        return self._f_mean(parents)

    def f(
        self, parents: dict[str, Tensor], u: Tensor, f_mean: Tensor | None = None
    ) -> Tensor:
        """Compute straight-through one-hot values from logits plus noise."""
        if f_mean is None:
            f_mean = self.f_mean(parents)

        v_circ = self._combine_logits_and_noise(f_mean, u)
        v_tilde = torch.softmax(v_circ, dim=-1)
        indices = torch.argmax(v_circ, dim=-1)
        v_bar = torch.nn.functional.one_hot(
            indices, num_classes=v_circ.shape[-1]
        ).to(v_tilde.dtype)
        return v_tilde + (v_bar - v_tilde).detach()

    @staticmethod
    def _combine_logits_and_noise(logits: Tensor, noise: Tensor) -> Tensor:
        """Broadcast/add noise to logits, supporting class-dimension expansion."""
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
