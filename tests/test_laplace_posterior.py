"""Tests for the LaplacePosterior class."""

import torch
from torch import Tensor
import pytest
from typing import Mapping
from inga.variable.base import Variable
from inga.variable.linear import LinearVariable
from inga.variable.functional import FunctionalVariable
from inga.approx_posterior.laplace import LaplacePosterior, LaplacePosteriorState
from inga.scm.base import SCM
from inga.scm.random import RandomSCMConfig, random_scm, resolve_transforms


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

    def test_init_exposes_map_and_optimizer_options(
        self, variables: dict[str, Variable]
    ) -> None:
        """Test constructor-based configuration of robust MAP options."""
        posterior = LaplacePosterior(
            variables=variables,
            continuation_scales=(2.0, 1.0),
            continuation_steps=7,
            num_map_restarts=4,
            restart_init_scales=(0.0, 0.3),
            num_mixture_components=2,
            adam_lr=1e-2,
            adam_scheduler_gamma=0.99,
            lbfgs_lr=0.8,
            lbfgs_max_iter=12,
            lbfgs_line_search_fn="strong_wolfe",
            jacobian_norm_cap=250.0,
        )

        assert posterior._continuation_scales == (2.0, 1.0)
        assert posterior._continuation_steps == 7
        assert posterior._num_map_restarts == 4
        assert posterior._restart_init_scales == (0.0, 0.3)
        assert posterior._num_mixture_components == 2
        assert posterior._adam_lr == 1e-2
        assert posterior._adam_scheduler_gamma == 0.99
        assert posterior._lbfgs_lr == 0.8
        assert posterior._lbfgs_max_iter == 12
        assert posterior._lbfgs_line_search_fn == "strong_wolfe"
        assert posterior._jacobian_norm_cap == 250.0

    def test_configure_map_options_updates_existing_posterior(
        self, posterior: LaplacePosterior
    ) -> None:
        """Test runtime update of robust MAP options."""
        posterior.configure_map_options(
            continuation_scales=(4.0, 2.0, 1.0),
            continuation_steps=9,
            num_map_restarts=5,
            restart_init_scales=(0.0, 0.2, 0.8),
            num_mixture_components=3,
            adam_lr=2e-2,
            adam_scheduler_gamma=0.98,
            lbfgs_lr=0.5,
            lbfgs_max_iter=20,
            lbfgs_line_search_fn="strong_wolfe",
            jacobian_norm_cap=500.0,
        )

        assert posterior._continuation_scales == (4.0, 2.0, 1.0)
        assert posterior._continuation_steps == 9
        assert posterior._num_map_restarts == 5
        assert posterior._restart_init_scales == (0.0, 0.2, 0.8)
        assert posterior._num_mixture_components == 3
        assert posterior._adam_lr == 2e-2
        assert posterior._adam_scheduler_gamma == 0.98
        assert posterior._lbfgs_lr == 0.5
        assert posterior._lbfgs_max_iter == 20
        assert posterior._jacobian_norm_cap == 500.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"continuation_scales": ()},
            {"continuation_scales": (1.0, 0.0)},
            {"continuation_steps": 0},
            {"num_map_restarts": 0},
            {"restart_init_scales": (-0.1,)},
            {"num_mixture_components": 0},
            {"adam_lr": 0.0},
            {"adam_scheduler_gamma": 1.5},
            {"lbfgs_lr": 0.0},
            {"lbfgs_max_iter": 0},
            {"lbfgs_line_search_fn": "invalid"},
            {"jacobian_norm_cap": 0.0},
        ],
    )
    def test_configure_map_options_validates_inputs(
        self, posterior: LaplacePosterior, kwargs: dict
    ) -> None:
        """Test invalid robust option configurations raise ValueError."""
        with pytest.raises(ValueError):
            posterior.configure_map_options(**kwargs)


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
        assert isinstance(state.MAP_components_rav, Tensor)
        assert isinstance(state.L_cov_components_rav, Tensor)
        assert isinstance(state.component_log_weights, Tensor)
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
        """Test that mixture MAP components have correct shape."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        # 10 samples, 1 component by default, 2 latent variables (Z, Y)
        assert state.MAP_components_rav.shape == (10, 1, 2)

    def test_fit_cov_shape(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that covariance Cholesky components have correct shape."""
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        # 10 samples, 1 component, 2x2 covariance matrix
        assert state.L_cov_components_rav.shape == (10, 1, 2, 2)

    def test_fit_with_multiple_observations(
        self, posterior: LaplacePosterior, observed_xy: dict[str, Tensor]
    ) -> None:
        """Test fit with multiple observed variables."""
        posterior.fit(observed_xy)
        state = posterior.state

        assert state is not None
        # Only Z is latent when X and Y are observed
        assert state.latent_names == ["Z"]
        assert state.MAP_components_rav.shape == (10, 1, 1)

    def test_fit_raises_with_empty_observed(self, posterior: LaplacePosterior) -> None:
        """Test that fit requires at least one observed variable."""
        with pytest.raises(ValueError, match="observed"):
            posterior.fit({})

    def test_fit_with_all_observed_variables_sets_empty_latent_state(
        self, posterior: LaplacePosterior, observed_xy: dict[str, Tensor]
    ) -> None:
        """Test fit behavior when no latent variables remain."""
        observed_all = {
            "Z": torch.randn_like(observed_xy["X"]),
            "X": observed_xy["X"],
            "Y": observed_xy["Y"],
        }

        posterior.fit(observed_all)
        state = posterior.state

        assert state is not None
        assert state.latent_names == []
        assert state.MAP_components_rav.shape == (10, 1, 0)
        assert state.L_cov_components_rav.shape == (10, 1, 0, 0)

    def test_fit_mixture_components_shapes_and_weights(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test optional multi-component state shapes and normalized weights."""
        posterior._num_mixture_components = 2
        posterior.fit(observed_x)
        state = posterior.state

        assert state is not None
        assert state.MAP_components_rav.shape == (10, 2, 2)
        assert state.L_cov_components_rav.shape == (10, 2, 2, 2)
        assert state.component_log_weights.shape == (10, 2)
        assert torch.allclose(
            torch.logsumexp(state.component_log_weights, dim=1),
            torch.zeros(10),
            atol=1e-6,
        )


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

    def test_sample_with_no_latent_variables_returns_empty_dict(
        self, posterior: LaplacePosterior, observed_xy: dict[str, Tensor]
    ) -> None:
        """Test sampling behavior when all SCM variables are observed."""
        observed_all = {
            "Z": torch.randn_like(observed_xy["X"]),
            "X": observed_xy["X"],
            "Y": observed_xy["Y"],
        }

        posterior.fit(observed_all)
        samples = posterior.sample(10)

        assert samples == {}


class TestLaplacePosteriorState:
    """Tests for LaplacePosteriorState dataclass."""

    def test_state_dataclass(self) -> None:
        """Test that LaplacePosteriorState is a valid dataclass."""
        state = LaplacePosteriorState(
            MAP_components_rav=torch.zeros(5, 1, 2),
            L_cov_components_rav=torch.eye(2)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(5, 1, -1, -1),
            component_log_weights=torch.zeros(5, 1),
            latent_names=["Z", "Y"],
        )

        assert state.MAP_components_rav.shape == (5, 1, 2)
        assert state.L_cov_components_rav.shape == (5, 1, 2, 2)
        assert state.component_log_weights.shape == (5, 1)
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

    def test_posterior_loss_fn_returns_per_sample_values(
        self, posterior: LaplacePosterior, observed_x: dict[str, Tensor]
    ) -> None:
        """Test that posterior loss is computed per observation sample."""
        u_latent = {
            "Z": torch.zeros_like(observed_x["X"]),
            "Y": torch.zeros_like(observed_x["X"]),
        }

        loss = posterior._posterior_loss_fn(u_latent, observed_x)

        assert loss.shape == (len(observed_x["X"]),)
        assert torch.all(loss >= 0)


class TestLaplacePosteriorRobustMapComponents:
    """Tests for continuation and multi-start robustness components."""

    def test_optimize_map_candidate_uses_continuation_schedule(
        self,
        posterior: LaplacePosterior,
        observed_x: dict[str, Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that continuation scales are actually used during candidate optimization."""
        posterior._continuation_scales = (3.0, 1.5, 1.0)
        posterior._continuation_steps = 2

        recorded_scales: list[float] = []

        def fake_loss(
            self: LaplacePosterior,
            u_latent: Mapping[str, Tensor],
            observed: dict[str, Tensor],
            obs_sigma_scale: float = 1.0,
        ) -> Tensor:
            recorded_scales.append(obs_sigma_scale)
            # Keep autograd path for optimizer steps.
            zero = torch.zeros_like(next(iter(u_latent.values())))
            return sum((u**2 for u in u_latent.values()), start=zero)

        monkeypatch.setattr(LaplacePosterior, "_posterior_loss_fn", fake_loss)

        reference = observed_x["X"]
        posterior._optimize_map_candidate(
            observed=observed_x,
            latent_names=["Z", "Y"],
            size=len(reference),
            reference=reference,
            init_scale=0.0,
        )

        # Continuation part should start with exactly this schedule pattern.
        assert recorded_scales[:6] == [3.0, 3.0, 1.5, 1.5, 1.0, 1.0]

    def test_fit_map_selects_best_restart_per_observation(
        self, posterior: LaplacePosterior
    ) -> None:
        """Test that multi-start selection is done independently for each row."""
        observed = {"X": torch.zeros(4)}
        latent_names = ["Z", "Y"]

        candidates = [
            {"Z": torch.zeros(4), "Y": torch.zeros(4)},
            {"Z": torch.ones(4), "Y": torch.ones(4)},
            {"Z": 2.0 * torch.ones(4), "Y": 2.0 * torch.ones(4)},
        ]
        losses = [
            torch.tensor([3.0, 1.0, 4.0, 8.0]),
            torch.tensor([2.0, 5.0, 0.0, 1.0]),
            torch.tensor([0.5, 2.0, 6.0, 3.0]),
        ]

        call_idx = {"i": 0}

        def fake_optimize(
            observed: dict[str, Tensor],
            latent_names: list[str],
            size: int,
            reference: Tensor,
            init_scale: float,
        ) -> dict[str, Tensor]:
            i = call_idx["i"]
            call_idx["i"] += 1
            return candidates[i]

        def fake_loss(
            u_latent: Mapping[str, Tensor],
            observed: dict[str, Tensor],
            obs_sigma_scale: float = 1.0,
        ) -> Tensor:
            marker = int(u_latent["Z"][0].item())
            return losses[marker]

        posterior._num_map_restarts = 3
        posterior._optimize_map_candidate = fake_optimize  # type: ignore[method-assign]
        posterior._posterior_loss_fn = fake_loss  # type: ignore[method-assign]

        best = posterior._fit_map(observed, latent_names)

        # Row-wise winners by losses are restarts: [2, 0, 1, 1].
        assert torch.allclose(best["Z"], torch.tensor([2.0, 0.0, 1.0, 1.0]))
        assert torch.allclose(best["Y"], torch.tensor([2.0, 0.0, 1.0, 1.0]))

    def test_multistart_objective_not_worse_than_single_start(
        self, variables: dict[str, Variable], observed_x: dict[str, Tensor]
    ) -> None:
        """Regression test: multi-start MAP should not worsen per-sample objective."""
        single = LaplacePosterior(variables=variables)
        multi = LaplacePosterior(variables=variables)

        # Keep test fast while preserving algorithmic behavior.
        for p in (single, multi):
            p._continuation_scales = (1.0,)
            p._continuation_steps = 8

        single._num_map_restarts = 1
        multi._num_map_restarts = 3

        torch.manual_seed(123)
        latent_names = [name for name in variables if name not in observed_x]
        map_single = single._fit_map(observed_x, latent_names)

        torch.manual_seed(123)
        map_multi = multi._fit_map(observed_x, latent_names)

        loss_single = single._posterior_loss_fn(map_single, observed_x)
        loss_multi = multi._posterior_loss_fn(map_multi, observed_x)

        assert torch.all(loss_multi <= loss_single + 1e-6)


class TestLaplacePosteriorNonlinearRobustness:
    """Stress tests for Laplace robustness on nonlinear/exponential SEMs."""

    @staticmethod
    def _assert_finite_and_bounded(t: Tensor, bound: float = 1e6) -> None:
        """Assert tensor values are finite and not explosively large."""
        assert torch.isfinite(t).all(), "Found non-finite values in tensor."
        assert t.abs().max() < bound, (
            f"Found excessively large values (max abs={t.abs().max().item():.3e})."
        )

    def test_nonlinear_exponential_model_fit_and_sample_are_numerically_stable(
        self,
    ) -> None:
        """Test a handcrafted nonlinear SCM with exp transforms for stability."""
        scm = SCM(
            variables=[
                FunctionalVariable(
                    name="Z",
                    sigma=0.8,
                    f_mean=lambda _: torch.tensor(0.0),
                    parent_names=[],
                ),
                FunctionalVariable(
                    name="X",
                    sigma=0.7,
                    f_mean=lambda p: torch.exp(0.25 * p["Z"]),
                    parent_names=["Z"],
                ),
                FunctionalVariable(
                    name="Y",
                    sigma=0.9,
                    f_mean=lambda p: 0.5 * torch.tanh(p["X"]) + 0.3 * torch.sin(p["Z"]),
                    parent_names=["X", "Z"],
                ),
                FunctionalVariable(
                    name="V",
                    sigma=1.0,
                    f_mean=lambda p: torch.exp(0.15 * p["Y"]),
                    parent_names=["Y"],
                ),
            ]
        )

        scm.posterior._num_map_restarts = 4
        scm.posterior._num_mixture_components = 2
        scm.posterior._continuation_scales = (4.0, 2.0, 1.0)
        scm.posterior._continuation_steps = 25

        torch.manual_seed(7)
        values = scm.generate(40)
        observed = {"X": values["X"], "V": values["V"]}

        scm.posterior.fit(observed)
        state = scm.posterior.state
        assert state is not None

        self._assert_finite_and_bounded(state.MAP_components_rav)
        self._assert_finite_and_bounded(state.L_cov_components_rav)
        self._assert_finite_and_bounded(state.component_log_weights, bound=100.0)

        torch.manual_seed(11)
        samples = scm.posterior.sample(200)
        for sample in samples.values():
            self._assert_finite_and_bounded(sample)

    def test_random_high_nonlinearity_models_remain_stable(self) -> None:
        """Test robustness across several random SEMs with nonlinear transforms."""
        seeds = [0, 1, 2]

        for seed in seeds:
            scm = random_scm(
                RandomSCMConfig(
                    num_variables=6,
                    parent_prob=0.65,
                    nonlinear_prob=1.0,
                    sigma_range=(0.7, 1.2),
                    coef_range=(-0.7, 0.7),
                    intercept_range=(-0.3, 0.3),
                    seed=seed,
                )
            )
            scm.posterior._num_map_restarts = 4
            scm.posterior._num_mixture_components = 2
            scm.posterior._continuation_scales = (3.0, 1.5, 1.0)
            scm.posterior._continuation_steps = 20

            torch.manual_seed(100 + seed)
            values = scm.generate(32)

            names = list(scm._variables.keys())
            observed_names = names[-2:]
            observed = {name: values[name] for name in observed_names}

            scm.posterior.fit(observed)
            state = scm.posterior.state
            assert state is not None

            self._assert_finite_and_bounded(state.MAP_components_rav)
            self._assert_finite_and_bounded(state.L_cov_components_rav)
            self._assert_finite_and_bounded(state.component_log_weights, bound=100.0)

            torch.manual_seed(200 + seed)
            samples = scm.posterior.sample(128)
            for sample in samples.values():
                self._assert_finite_and_bounded(sample)

    def test_laplace_stable_with_all_supported_transforms(self) -> None:
        """Ensure Laplace fitting remains stable when every transform is exercised."""
        transform_names = [
            "sin",
            "cos",
            "exp",
            "tanh",
            "sigmoid",
            "softsign",
            "atan",
            "swish",
            "gelu",
            "relu",
            "leaky_relu",
            "elu",
            "softplus_sharp",
            "abs",
            "cubic",
        ]
        transforms = resolve_transforms(transform_names)

        def all_transforms_chain(z: Tensor) -> Tensor:
            value = 0.35 * z
            for transform in transforms:
                value = transform(value)
                # Keep numerical range controlled while still preserving
                # participation of each transform in the computational graph.
                value = 0.5 * value
            return value

        scm = SCM(
            variables=[
                FunctionalVariable(
                    name="Z",
                    sigma=0.8,
                    f_mean=lambda _: torch.tensor(0.0),
                    parent_names=[],
                ),
                FunctionalVariable(
                    name="X",
                    sigma=0.7,
                    f_mean=lambda p: all_transforms_chain(p["Z"]),
                    parent_names=["Z"],
                ),
                FunctionalVariable(
                    name="Y",
                    sigma=0.8,
                    f_mean=lambda p: 0.4 * p["X"] + 0.2 * torch.tanh(p["Z"]),
                    parent_names=["X", "Z"],
                ),
            ]
        )

        scm.posterior._num_map_restarts = 3
        scm.posterior._num_mixture_components = 2
        scm.posterior._continuation_scales = (3.0, 1.5, 1.0)
        scm.posterior._continuation_steps = 18

        torch.manual_seed(321)
        values = scm.generate(32)
        observed = {"X": values["X"], "Y": values["Y"]}

        scm.posterior.fit(observed)
        state = scm.posterior.state
        assert state is not None
        self._assert_finite_and_bounded(state.MAP_components_rav)
        self._assert_finite_and_bounded(state.L_cov_components_rav)
        self._assert_finite_and_bounded(state.component_log_weights, bound=100.0)

        torch.manual_seed(322)
        samples = scm.posterior.sample(64)
        assert "Z" in samples
        self._assert_finite_and_bounded(samples["Z"])


class TestLaplacePosteriorJacobianStandardization:
    """Tests for Jacobian-norm standardization in GN Hessian."""

    def test_jacobian_norm_cap_controls_gn_curvature(
        self, variables: dict[str, Variable]
    ) -> None:
        """Smaller jacobian_norm_cap should yield larger covariance factors."""
        observed = {"X": torch.zeros(6)}
        latent_names = ["Z", "Y"]
        u_latent_rav = torch.zeros(6, 2)

        posterior_lo = LaplacePosterior(variables=variables, jacobian_norm_cap=1.0)
        posterior_hi = LaplacePosterior(variables=variables, jacobian_norm_cap=1e9)

        def fake_f_mean_u(
            u_latent_rav: Tensor,
            observed: dict[str, Tensor],
            observed_name: str,
            latent_names: list[str],
        ) -> Tensor:
            # Constant very large gradient wrt latent variables.
            return 1e8 * (u_latent_rav[0] + 0.5 * u_latent_rav[1])

        posterior_lo._f_mean_u = fake_f_mean_u  # type: ignore[method-assign]
        posterior_hi._f_mean_u = fake_f_mean_u  # type: ignore[method-assign]

        L_lo = posterior_lo._approx_cov_chol(u_latent_rav, observed, latent_names)
        L_hi = posterior_hi._approx_cov_chol(u_latent_rav, observed, latent_names)

        assert torch.isfinite(L_lo).all()
        assert torch.isfinite(L_hi).all()
        # Stronger cap (smaller value) means weaker GN curvature and thus
        # larger posterior covariance factor magnitude.
        assert L_lo.abs().mean() > L_hi.abs().mean()
