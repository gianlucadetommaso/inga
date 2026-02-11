"""Tests for the SEM class."""

import torch
import pytest
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM
from steindag.sem.random import RandomSEMConfig, _build_f_mean, random_sem


@pytest.fixture
def simple_sem() -> SEM:
    """Create a simple SEM with Z -> X structure."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
            ),
        ]
    )


@pytest.fixture
def chain_sem() -> SEM:
    """Create a chain SEM: Z -> X -> Y."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
            ),
            LinearVariable(
                name="Y", parent_names=["X"], sigma=1.0, coefs={"X": 2.0}, intercept=0.0
            ),
        ]
    )


@pytest.fixture
def collider_sem() -> SEM:
    """Create a collider SEM: Z -> Y <- X."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="Y",
                parent_names=["Z", "X"],
                sigma=1.0,
                coefs={"Z": 1.0, "X": 2.0},
                intercept=0.0,
            ),
        ]
    )


class TestSEMInit:
    """Tests for SEM initialization."""

    def test_init_stores_variables(self, simple_sem: SEM) -> None:
        """Test that __init__ correctly stores variables by name."""
        assert "Z" in simple_sem._variables
        assert "X" in simple_sem._variables
        assert len(simple_sem._variables) == 2

    def test_init_creates_posterior(self, simple_sem: SEM) -> None:
        """Test that __init__ creates a LaplacePosterior."""
        assert simple_sem.posterior is not None

    def test_init_passes_posterior_kwargs(self) -> None:
        """Test that SEM forwards posterior configuration to LaplacePosterior."""
        sem = SEM(
            variables=[
                LinearVariable(
                    name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
                LinearVariable(
                    name="X",
                    parent_names=["Z"],
                    sigma=1.0,
                    coefs={"Z": 1.0},
                    intercept=0.0,
                ),
            ],
            posterior_kwargs={
                "num_map_restarts": 4,
                "num_mixture_components": 2,
                "adam_lr": 1e-2,
            },
        )

        assert sem.posterior._num_map_restarts == 4
        assert sem.posterior._num_mixture_components == 2
        assert sem.posterior._adam_lr == 1e-2


class TestSEMGenerate:
    """Tests for SEM.generate method."""

    def test_generate_returns_all_variables(self, simple_sem: SEM) -> None:
        """Test that generate returns values for all variables."""
        torch.manual_seed(42)
        values = simple_sem.generate(10)

        assert "Z" in values
        assert "X" in values
        assert len(values) == 2

    def test_generate_correct_shape(self, simple_sem: SEM) -> None:
        """Test that generated values have correct shape."""
        torch.manual_seed(42)
        values = simple_sem.generate(100)

        assert values["Z"].shape == (100,)
        assert values["X"].shape == (100,)

    def test_generate_respects_causal_structure(self, chain_sem: SEM) -> None:
        """Test that generate respects the causal structure."""
        torch.manual_seed(42)
        values = chain_sem.generate(1000)

        # Y should be correlated with X (Y = 2*X + noise)
        # Correlation should be positive and significant
        corr = torch.corrcoef(torch.stack([values["X"], values["Y"]]))[0, 1]
        assert corr > 0.5

    def test_generate_reproducible_with_seed(self, simple_sem: SEM) -> None:
        """Test that generate is reproducible with the same seed."""
        torch.manual_seed(42)
        values1 = simple_sem.generate(10)

        torch.manual_seed(42)
        values2 = simple_sem.generate(10)

        assert torch.allclose(values1["Z"], values2["Z"])
        assert torch.allclose(values1["X"], values2["X"])

    def test_generate_collider_structure(self, collider_sem: SEM) -> None:
        """Test generate with collider structure (independent causes)."""
        torch.manual_seed(42)
        values = collider_sem.generate(1000)

        # Z and X should be approximately uncorrelated (independent causes)
        corr = torch.corrcoef(torch.stack([values["Z"], values["X"]]))[0, 1]
        assert abs(corr) < 0.1

    def test_generate_single_sample(self, simple_sem: SEM) -> None:
        """Test generate with a single sample."""
        torch.manual_seed(42)
        values = simple_sem.generate(1)

        assert values["Z"].shape == (1,)
        assert values["X"].shape == (1,)


class TestSEMPosterior:
    """Tests for SEM posterior inference."""

    def test_posterior_fit_and_sample(self, simple_sem: SEM) -> None:
        """Test that posterior.fit and posterior.sample work."""
        torch.manual_seed(42)
        values = simple_sem.generate(10)

        observed = {"X": values["X"]}
        simple_sem.posterior.fit(observed)

        torch.manual_seed(0)
        samples = simple_sem.posterior.sample(100)

        assert "Z" in samples
        assert samples["Z"].shape == (10, 100)

    def test_posterior_sample_requires_fit(self, simple_sem: SEM) -> None:
        """Test that sample raises error if fit not called."""
        # Create a fresh SEM to ensure no state
        sem = SEM(
            variables=[
                LinearVariable(
                    name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
            ]
        )

        with pytest.raises(ValueError, match="fit"):
            sem.posterior.sample(10)


class TestRandomSEM:
    """Tests for random SEM generation."""

    def test_random_sem_generates_dag(self) -> None:
        """Ensure random SEM respects DAG ordering."""
        config = RandomSEMConfig(num_variables=6, parent_prob=0.6, seed=123)
        sem = random_sem(config)
        assert len(sem._variables) == 6

        for idx, (name, variable) in enumerate(sem._variables.items()):
            allowed_parents = {f"X{i}" for i in range(idx)}
            assert set(variable.parent_names).issubset(allowed_parents)

    def test_random_sem_reproducible(self) -> None:
        """Ensure the random SEM is reproducible with a seed."""
        config = RandomSEMConfig(num_variables=4, parent_prob=0.5, seed=99)
        sem1 = random_sem(config)
        sem2 = random_sem(config)

        for (name1, var1), (name2, var2) in zip(
            sem1._variables.items(), sem2._variables.items()
        ):
            assert name1 == name2
            assert var1.parent_names == var2.parent_names

    def test_random_sem_rejects_invalid_prob(self) -> None:
        """Ensure invalid parent probabilities raise errors."""
        with pytest.raises(ValueError, match="parent_prob"):
            random_sem(RandomSEMConfig(num_variables=3, parent_prob=1.5))

    def test_random_sem_rejects_non_positive_size(self) -> None:
        """Ensure invalid num_variables raises errors."""
        with pytest.raises(ValueError, match="num_variables"):
            random_sem(RandomSEMConfig(num_variables=0))

    def test_random_sem_includes_nonlinear_variables(self) -> None:
        """Ensure nonlinear option creates at least one FunctionalVariable when forced."""
        from steindag.variable.functional import FunctionalVariable

        config = RandomSEMConfig(
            num_variables=4,
            parent_prob=0.7,
            nonlinear_prob=1.0,
            seed=7,
        )
        sem = random_sem(config)
        assert any(
            isinstance(var, FunctionalVariable) for var in sem._variables.values()
        )

    def test_build_f_mean_normalizes_linear_predictor_gradient(self) -> None:
        """Linear pre-activation gradient should be coefficient-norm standardized."""
        f_mean = _build_f_mean(
            parent_names=["A", "B"],
            coefs={"A": 3.0, "B": 4.0},
            intercept=0.0,
            transforms=None,
        )

        a = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        out = f_mean({"A": a, "B": b})

        da, db = torch.autograd.grad(out, [a, b])
        grad_norm = torch.sqrt(da**2 + db**2)

        expected_norm = torch.tensor((25.0 / 26.0) ** 0.5)
        assert torch.allclose(grad_norm, expected_norm, atol=1e-6)
        assert torch.allclose(da, torch.tensor(3.0 / (26.0**0.5)), atol=1e-6)
        assert torch.allclose(db, torch.tensor(4.0 / (26.0**0.5)), atol=1e-6)

    def test_random_sem_seed22_generate_is_finite_regression(self) -> None:
        """Regression: seed-22 random SEM generation should not emit NaN/Inf."""
        sem = random_sem(
            RandomSEMConfig(
                num_variables=6,
                parent_prob=0.6,
                nonlinear_prob=0.8,
                sigma_range=(0.7, 1.2),
                coef_range=(-1.0, 1.0),
                intercept_range=(-0.5, 0.5),
                seed=22,
            )
        )
        data = sem.generate(768)
        for values in data.values():
            assert torch.isfinite(values).all()
