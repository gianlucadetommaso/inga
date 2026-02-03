import torch
from torch import Tensor
import pytest
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM


@pytest.fixture
def sem() -> SEM:
    return SEM(
        variables=[
            LinearVariable(name="Z", parent_names=[], sigma=1., coefs={}, intercept=0.),
            LinearVariable(name="X", parent_names=["Z"], sigma=1., coefs={"Z": 1.}, intercept=0.),
            LinearVariable(name="Y", parent_names=["X", "Z"], sigma=1., coefs={"X": 2., "Z": 3.}, intercept=0.)
        ]
    )


@pytest.fixture
def values(sem: SEM) -> dict[str, Tensor]:
    torch.manual_seed(42)
    return sem.generate(10)


class TestPosteriorStats:
    def test_posterior_stats_observe_x_only(self, sem: SEM, values: dict[str, Tensor]) -> None:
        """Test posterior mean and variance when only X is observed."""
        alpha: float = sem._variables["X"]._coefs["Z"]
        
        observed: dict[str, Tensor] = {"X": values["X"]}
        
        maps: dict[str, Tensor] = sem.fit_map(observed)
        chols_cov: Tensor = sem.approx_cov_chol(maps, observed)
        maps_rav: Tensor = sem._ravel(maps)
        
        torch.manual_seed(0)
        samples: dict[str, Tensor] = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))
        
        expected_means: Tensor = alpha * observed["X"] / (1 + alpha ** 2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha ** 2)]).expand(10)
        
        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))
        
        assert torch.all(mean_error < 0.05), f"Posterior mean error too large: {mean_error}"
        assert torch.all(var_error < 0.05), f"Posterior variance error too large: {var_error}"
    
    def test_posterior_stats_observe_x_and_y(self, sem: SEM, values: dict[str, Tensor]) -> None:
        """Test posterior mean and variance when X and Y are observed."""
        alpha: float = sem._variables["X"]._coefs["Z"]
        beta: float = sem._variables["Y"]._coefs["X"]
        gamma: float = sem._variables["Y"]._coefs["Z"]
        
        observed: dict[str, Tensor] = {"X": values["X"], "Y": values["Y"]}
        
        maps: dict[str, Tensor] = sem.fit_map(observed)
        chols_cov: Tensor = sem.approx_cov_chol(maps, observed)
        maps_rav: Tensor = sem._ravel(maps)
        
        torch.manual_seed(0)
        samples: dict[str, Tensor] = sem.sample(maps_rav, chols_cov, 10000, sem._get_latent_names(maps))
        
        expected_means: Tensor = (gamma * (observed["Y"] - beta * observed["X"]) + alpha * observed["X"]) / (1 + alpha ** 2 + gamma ** 2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha ** 2 + gamma ** 2)]).expand(10)
        
        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))
        
        assert torch.all(mean_error < 0.05), f"Posterior mean error too large: {mean_error}"
        assert torch.all(var_error < 0.05), f"Posterior variance error too large: {var_error}"
