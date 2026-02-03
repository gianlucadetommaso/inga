"""Tests for the LaplacePosterior class."""

import torch
from torch import Tensor
import pytest
from steindag.variable.base import Variable
from steindag.variable.linear import LinearVariable
from steindag.approx_posterior.laplace import LaplacePosterior, LaplacePosteriorState


@pytest.fixture
def variables() -> dict[str, Variable]:
    """Create a dict of variables for testing."""
    return {
        "Z": LinearVariable(
            name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
        ),
        "X": LinearVariable(
            name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
        ),
        "Y": LinearVariable(
            name="Y",
            parent_names=["X", "Z"],
            sigma=1.0,
            coefs={"X": 2.0, "Z": 3.0},
            intercept=0.0,
        ),
    }


@pytest.fixture
def posterior(variables: dict[str, Variable]) -> LaplacePosterior:
    """Create a LaplacePosterior for testing."""
    return LaplacePosterior(variables=variables)


@pytest.fixture
def observed_x(variables: dict[str, Variable]) -> dict[str, Tensor]:
    """Generate observed data for X only."""
    torch.manual_seed(42)
    # Simple forward sampling
    z = torch.randn(10)
    x = z + torch.randn(10)  # X = Z + noise
    return {"X": x}


@pytest.fixture
def observed_xy(variables: dict[str, Variable]) -> dict[str, Tensor]:
    """Generate observed data for X and Y."""
    torch.manual_seed(42)
    z = torch.randn(10)
    x = z + torch.randn(10)
    y = 2 * x + 3 * z + torch.randn(10)
    return {"X": x, "Y": y}


class TestLaplacePosteriorInit:
    """Tests for LaplacePosterior initialization."""

    def test_init_stores_variables(
        self, posterior: LaplacePosterior, variables: dict[str, Variable]
    ) -> None:
        """Test that __init__ stores variables."""
        assert posterior._variables == variables

    def test_init_state_is_none(self, posterior: LaplacePosterior) -> None:
        """Test that state is None before fit."""
        assert posterior.state is None


class TestLaplacePosteriorFit:
    """Tests for LaplacePosterior.fit method."""

    def test_fit_sets_state(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that fit sets the state."""
        posterior.fit(observed_x)

        assert posterior.state is not None
        assert isinstance(posterior.state, LaplacePosteriorState)

    def test_fit_state_has_correct_attributes(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that fitted state has correct attributes."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        assert isinstance(state.MAP_rav, Tensor)
        assert isinstance(state.L_cov_rav, Tensor)
        assert isinstance(state.latent_names, list)

    def test_fit_latent_names_excludes_observed(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that latent_names excludes observed variables."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        assert "X" not in state.latent_names
        assert "Z" in state.latent_names
        assert "Y" in state.latent_names

    def test_fit_map_shape(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that MAP has correct shape."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        # 10 samples, 2 latent variables (Z, Y)
        assert state.MAP_rav.shape == (10, 2)

    def test_fit_cov_shape(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that covariance Cholesky has correct shape."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        # 10 samples, 2x2 covariance matrix
        assert state.L_cov_rav.shape == (10, 2, 2)

    def test_fit_with_multiple_observations(
        self, posterior: LaplacePosterior, observed_xy: dict[str, Tensor]
    ) -> None:
        """Test fit with multiple observed variables."""
        posterior.fit(observed_xy)
        state = posterior.state

        assert state is not None
        # Only Z is latent when X and Y are observed
        assert state.latent_names == ["Z"]
        assert state.MAP_rav.shape == (10, 1)


class TestLaplacePosteriorSample:
    """Tests for LaplacePosterior.sample method."""

    def test_sample_requires_fit(self, posterior: LaplacePosterior) -> None:
        """Test that sample raises error if fit not called."""
        with pytest.raises(ValueError, match="fit"):
            posterior.sample(10)

    def test_sample_returns_dict(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that sample returns a dictionary."""
        posterior.fit(observed_x)
        samples = posterior.sample(100)

        assert isinstance(samples, dict)

    def test_sample_contains_latent_variables(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that sample contains all latent variables."""
        posterior.fit(observed_x)
        samples = posterior.sample(100)

        assert "Z" in samples
        assert "Y" in samples
        assert "X" not in samples  # X is observed

    def test_sample_correct_shape(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that samples have correct shape."""
        posterior.fit(observed_x)
        samples = posterior.sample(100)

        # 10 data points, 100 samples each
        assert samples["Z"].shape == (10, 100)
        assert samples["Y"].shape == (10, 100)

    def test_sample_reproducible_with_seed(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that sample is reproducible with same seed."""
        posterior.fit(observed_x)

        torch.manual_seed(42)
        samples1 = posterior.sample(100)

        torch.manual_seed(42)
        samples2 = posterior.sample(100)

        assert torch.allclose(samples1["Z"], samples2["Z"])

    def test_sample_variability(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that samples have non-zero variance."""
        posterior.fit(observed_x)
        samples = posterior.sample(1000)

        # Samples should have some variance
        assert samples["Z"].var(dim=1).mean() > 0.01


class TestLaplacePosteriorState:
    """Tests for LaplacePosteriorState dataclass."""

    def test_state_dataclass(self) -> None:
        """Test that LaplacePosteriorState is a valid dataclass."""
        state = LaplacePosteriorState(
            MAP_rav=torch.zeros(5, 2),
            L_cov_rav=torch.eye(2).unsqueeze(0).expand(5, -1, -1),
            latent_names=["Z", "Y"],
        )

        assert state.MAP_rav.shape == (5, 2)
        assert state.L_cov_rav.shape == (5, 2, 2)
        assert state.latent_names == ["Z", "Y"]


class TestLaplacePosteriorHelperMethods:
    """Tests for LaplacePosterior helper methods."""

    def test_get_latent_names(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test _get_latent_names returns correct order."""
        posterior.fit(observed_x)
        state = posterior.state
        assert state is not None

        # Should be in topological order (Z before Y)
        assert state.latent_names.index("Z") < state.latent_names.index("Y")

    def test_ravel_unravel_roundtrip(self, posterior: LaplacePosterior) -> None:
        """Test that ravel and unravel are inverses."""
        u_latent = {
            "Z": torch.tensor([1.0, 2.0, 3.0]),
            "Y": torch.tensor([4.0, 5.0, 6.0]),
        }

        raveled = posterior._ravel(u_latent)
        latent_names = posterior._get_latent_names(u_latent)
        unraveled = posterior._unravel(raveled, latent_names)

        assert torch.allclose(u_latent["Z"], unraveled["Z"])
        assert torch.allclose(u_latent["Y"], unraveled["Y"])
