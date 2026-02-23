"""Linear variable implementation for structural causal models."""

import torch
from torch import Tensor
from typing import Iterable
from inga.scm.variable.base import GaussianVariable


class LinearVariable(GaussianVariable):
    """A variable with a linear mean function.

    The mean function is: intercept + sum(coefs[parent] * parent for parent in parents).

    Attributes:
        name: The variable's identifier.
        parent_names: Names of parent variables in the DAG.
        sigma: Standard deviation of the additive noise term.
    """

    def __init__(
        self,
        name: str,
        sigma: float,
        coefs: dict[str, float] | None = None,
        intercept: float = 0.0,
        parent_names: Iterable[str] | None = None,
    ) -> None:
        """Initialize a linear variable.

        Args:
            name: The variable's identifier.
            sigma: Standard deviation of the additive noise term.
            coefs: Dictionary mapping parent names to their linear coefficients.
                Defaults to empty dict.
            intercept: The constant term in the linear function. Defaults to 0.0.
            parent_names: Names of parent variables in the DAG. If None, inferred
                from coefs keys.

        Raises:
            ValueError: If coefs keys don't match parent_names.
        """
        # Default coefs to empty dict
        if coefs is None:
            coefs = {}

        # Infer parent_names from coefs keys if not provided
        if parent_names is None:
            parent_names = list(coefs.keys())
        else:
            parent_names = list(parent_names)

        # Validate that coefs keys match parent_names
        coef_keys = set(coefs.keys())
        parent_set = set(parent_names)
        if coef_keys != parent_set:
            missing_in_coefs = parent_set - coef_keys
            extra_in_coefs = coef_keys - parent_set
            msg_parts = []
            if missing_in_coefs:
                msg_parts.append(f"missing coefficients for: {missing_in_coefs}")
            if extra_in_coefs:
                msg_parts.append(f"extra coefficients for: {extra_in_coefs}")
            raise ValueError(
                f"Coefficient keys must match parent_names. {'; '.join(msg_parts)}"
            )

        super().__init__(name, sigma, parent_names)

        self._coefs = coefs
        self._intercept = intercept

    def f_mean(self, parents: dict[str, Tensor]) -> Tensor:
        """Compute the linear mean function.

        Args:
            parents: Dictionary mapping parent names to their tensor values.

        Returns:
            The linear combination: intercept + sum(coefs[parent] * parent).
        """
        if not parents:
            return torch.tensor(self._intercept)

        f_mean: Tensor | float = self._intercept
        for parent_name, parent in parents.items():
            f_mean = f_mean + self._coefs[parent_name] * parent

        return f_mean  # type: ignore[return-value]

    def f(
        self,
        parents: dict[str, Tensor],
        u: Tensor,
        f_mean: Tensor | None = None,
    ) -> Tensor:
        """Compute variable values as linear mean plus scaled Gaussian noise."""
        return super().f(parents=parents, u=u, f_mean=f_mean)
