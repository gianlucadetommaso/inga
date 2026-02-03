"""Tests for posterior statistics of the SEM.

These tests verify that the Laplace approximate posterior inference produces
correct mean and variance estimates by comparing against analytically
derived ground truth for a simple linear SEM.
"""

import torch
from torch import Tensor
import pytest
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM


@pytest.fixture
def sem() -> SEM:
    """Create a test SEM with structure Z -> X, Z -> Y, X -> Y.

    Returns:
        A SEM with three linear variables and unit noise.
    """
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
            ),
            LinearVariable(
                name="Y",
                parent_names=["X", "Z"],
                sigma=1.0,
                coefs={"X": 2.0, "Z": 3.0},
                intercept=0.0,
            ),
        ]
    )


@pytest.fixture
def values(sem: SEM) -> dict[str, Tensor]:
    """Generate sample values from the SEM.

    Args:
        sem: The structural equation model.

    Returns:
        Dictionary of generated values for all variables.
    """
    torch.manual_seed(42)
    return sem.generate(10)


def _get_coef(sem: SEM, var_name: str, parent_name: str) -> float:
    """Get a linear coefficient from a LinearVariable in the SEM.

    Args:
        sem: The structural equation model.
        var_name: Name of the variable.
        parent_name: Name of the parent variable.

    Returns:
        The linear coefficient for the specified parent.

    Raises:
        AssertionError: If the variable is not a LinearVariable.
    """
    var = sem._variables[var_name]
    assert isinstance(var, LinearVariable)
    return var._coefs[parent_name]


class TestPosteriorStats:
    """Tests for posterior mean and variance accuracy using Laplace approximation."""

    def test_posterior_stats_observe_x_only(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test posterior mean and variance when only X is observed.

        For the model Z -> X with X = alpha*Z + noise, the analytical
        posterior of Z given X is:
        - mean: alpha * X / (1 + alpha^2)
        - variance: 1 / (1 + alpha^2)
        """
        alpha = _get_coef(sem, "X", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}

        sem.posterior.fit(observed)

        torch.manual_seed(0)
        samples = sem.posterior.sample(10000)

        expected_means: Tensor = alpha * observed["X"] / (1 + alpha**2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha**2)]).expand(10)

        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))

        assert torch.all(mean_error < 0.05), (
            f"Posterior mean error too large: {mean_error}"
        )
        assert torch.all(var_error < 0.05), (
            f"Posterior variance error too large: {var_error}"
        )

    def test_posterior_stats_observe_x_and_y(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test posterior mean and variance when X and Y are observed.

        For the model Z -> X, Z -> Y, X -> Y with observations blocking
        the X -> Y path, the analytical posterior of Z given X, Y is:
        - mean: (gamma * (Y - beta*X) + alpha * X) / (1 + alpha^2 + gamma^2)
        - variance: 1 / (1 + alpha^2 + gamma^2)
        """
        alpha = _get_coef(sem, "X", "Z")
        beta = _get_coef(sem, "Y", "X")
        gamma = _get_coef(sem, "Y", "Z")

        observed: dict[str, Tensor] = {"X": values["X"], "Y": values["Y"]}

        sem.posterior.fit(observed)

        torch.manual_seed(0)
        samples = sem.posterior.sample(10000)

        expected_means: Tensor = (
            gamma * (observed["Y"] - beta * observed["X"]) + alpha * observed["X"]
        ) / (1 + alpha**2 + gamma**2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha**2 + gamma**2)]).expand(10)

        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))

        assert torch.all(mean_error < 0.05), (
            f"Posterior mean error too large: {mean_error}"
        )
        assert torch.all(var_error < 0.05), (
            f"Posterior variance error too large: {var_error}"
        )
