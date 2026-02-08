"""Tests for posterior statistics of the SEM.

These tests verify that the Laplace approximate posterior inference produces
correct mean and variance estimates by comparing against analytically
derived ground truth for a simple linear SEM.
"""

import torch
from torch import Tensor
from torch.func import grad
from functools import partial
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
        samples = sem.posterior.sample(2000)

        expected_means: Tensor = alpha * observed["X"] / (1 + alpha**2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha**2)]).expand(10)

        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))

        assert torch.all(mean_error < 0.08), (
            f"Posterior mean error too large: {mean_error}"
        )
        assert torch.all(var_error < 0.08), (
            f"Posterior variance error too large: {var_error}"
        )


class TestCausalBiasComponents:
    """Tests for individual components of the causal bias formula for linear confounder case.

    For the linear model Z -> X -> Y with Z -> Y (confounder), when only X is observed:
    - du_fy = 0 (dY/du_X = 0 because Y depends on X, not u_X directly)
    - dx_diff = -1 (for treatment variable)
    - mid_term = gamma * alpha / (1 + alpha^2) (confounding bias contribution)
    """

    def test_du_fy_is_zero_for_treatment(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test that du_fy = 0 for the treatment variable X.

        For Y = beta*X + gamma*Z + u_Y, the derivative dY/du_X should be 0
        because Y's structural equation depends on X directly, not through u_X.
        """

        observed: dict[str, Tensor] = {"X": values["X"][:1]}
        sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = sem.posterior.sample(1)
        u_latent = {k: v[0, 0] for k, v in u_latent_samples.items()}
        obs = {k: v[0] for k, v in observed.items()}

        u_x = sem._get_u_observed(u_latent, obs, "X")

        f_mean_u = partial(
            sem._compute_outcome_mean_from_noise,
            latent=u_latent,
            observed=obs,
            input_name="X",
            treatment_name="X",
            outcome_name="Y",
        )
        du_fy = grad(f_mean_u)(u_x)

        assert torch.allclose(du_fy, torch.tensor(0.0), atol=1e-5), (
            f"du_fy should be 0 for treatment variable, got {du_fy}"
        )

    def test_mid_term_equals_expected_for_linear_confounder(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test that mean of mid_term equals gamma*alpha/(1+alpha^2).

        The mid_term captures the confounding bias contribution, and its
        expected value should equal gamma * alpha / (1 + alpha^2).
        """
        alpha = _get_coef(sem, "X", "Z")
        gamma = _get_coef(sem, "Y", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = sem.posterior.sample(2000)
        outcome_means = sem._compute_conditional_outcome_mean(
            u_latent_samples, observed, outcome_name="Y"
        )

        # Compute mid_term for each sample and observation, then average
        mid_terms = []
        num_obs = observed["X"].shape[0]
        num_samples = 2000

        for i in range(num_obs):
            sample_mid_terms = []
            for j in range(num_samples):
                u_latent = {k: v[i, j] for k, v in u_latent_samples.items()}
                obs = {k: v[i] for k, v in observed.items()}
                mid_term = sem._compute_bias_mid_term(
                    u_latent,
                    obs,
                    outcome_mean=outcome_means[i],
                    observed_name="X",
                    outcome_name="Y",
                )
                sample_mid_terms.append(mid_term)
            mid_terms.append(torch.stack(sample_mid_terms).mean())

        mean_mid_term = torch.stack(mid_terms).mean()
        expected_mid_term = gamma * alpha / (1 + alpha**2)

        assert torch.allclose(
            mean_mid_term, torch.tensor(expected_mid_term), atol=0.15
        ), (
            f"mid_term mean should equal gamma*alpha/(1+alpha^2)={expected_mid_term}, got {mean_mid_term}"
        )

    def test_causal_effect_equals_beta_when_observing_x_only(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test that causal effect of X on Y equals beta when only X is observed.

        For the linear model Y = beta*X + gamma*Z + noise, the causal effect
        of X on Y is the partial derivative dY/dX = beta, which is constant
        across all observations.

        The variance of the causal effect should be 0 since beta is a constant.
        """
        beta = _get_coef(sem, "Y", "X")

        observed: dict[str, Tensor] = {"X": values["X"]}

        sem.posterior.fit(observed)

        causal_effect = sem.causal_effect(
            observed, treatment_name="X", outcome_name="Y"
        )
        causal_effect_var = sem.causal_effect_var(
            observed, treatment_name="X", outcome_name="Y"
        )

        expected_causal_effect = torch.full_like(causal_effect, beta)
        expected_causal_effect_var = torch.zeros_like(causal_effect_var)

        assert torch.allclose(causal_effect, expected_causal_effect, atol=1e-5), (
            f"Causal effect should equal beta={beta}, got {causal_effect}"
        )
        assert torch.allclose(
            causal_effect_var, expected_causal_effect_var, atol=1e-5
        ), f"Causal effect variance should be 0, got {causal_effect_var}"

    def test_causal_bias_when_observing_x_only(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test that causal bias of X on Y equals gamma*alpha/(1+alpha^2) when only X is observed.

        For the linear model with Z -> X -> Y and Z -> Y, when only X is observed,
        the causal bias arises from the confounding path through Z.
        The analytical causal bias is: gamma * alpha / (1 + alpha^2)

        where:
        - alpha is the coefficient from Z to X
        - gamma is the coefficient from Z to Y
        """
        alpha = _get_coef(sem, "X", "Z")
        gamma = _get_coef(sem, "Y", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}

        sem.posterior.fit(observed)

        causal_bias = sem.causal_bias(observed, treatment_name="X", outcome_name="Y")

        expected_causal_bias = torch.full_like(
            causal_bias, gamma * alpha / (1 + alpha**2)
        )

        assert torch.allclose(causal_bias, expected_causal_bias, atol=0.2), (
            f"Causal bias should equal gamma*alpha/(1+alpha^2)={gamma * alpha / (1 + alpha**2)}, got {causal_bias}"
        )

    def test_causal_regularization_is_ratio_of_sample_means(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test causal_regularization equals ratio of numerator/denominator sample means."""
        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)

        torch.manual_seed(0)
        latent_samples = sem.posterior.sample(256)
        num_samples = sem._compute_causal_regularization_numerator_term_samples(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            observed_name="X",
            latent_samples=latent_samples,
        )
        den_samples = sem._compute_causal_regularization_denominator_term_samples(
            observed=observed,
            treatment_name="X",
            observed_name="X",
            latent_samples=latent_samples,
        )

        torch.manual_seed(0)
        reg = sem.causal_regularization(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            num_samples=256,
        )

        assert torch.allclose(
            reg, num_samples.mean(dim=1) / den_samples.mean(dim=1), atol=2e-2
        )

    def test_causal_regularization_samples_differ_from_bias_samples(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test that regularization samples are not identical to bias samples."""
        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)

        torch.manual_seed(0)
        bias_samples = sem._compute_causal_bias_samples(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            num_samples=256,
        )

        torch.manual_seed(0)
        reg_samples = sem._compute_causal_regularization_samples(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            num_samples=256,
        )

        assert not torch.allclose(bias_samples, reg_samples)

    def test_causal_regularization_matches_linear_confounder_closed_form(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test closed form causal regularizer for the linear confounder model.

        For Z -> X, Z -> Y, X -> Y with only X observed, verifies:
            r(x) = (beta + gamma*alpha/(1+alpha^2)) * x - gamma*alpha/x
        """
        alpha = _get_coef(sem, "X", "Z")
        beta = _get_coef(sem, "Y", "X")
        gamma = _get_coef(sem, "Y", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)

        reg = sem.causal_regularization(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            num_samples=2000,
        )

        x = observed["X"]
        expected = (beta + gamma * alpha / (1 + alpha**2)) * x - gamma * alpha / x

        stable_mask = torch.abs(x) > 0.5

        assert torch.allclose(reg[stable_mask], expected[stable_mask], atol=0.35), (
            "Causal regularization should match closed form "
            "(beta + gamma*alpha/(1+alpha^2))*x - gamma*alpha/x"
        )

    def test_causal_regularization_numerator_matches_closed_form(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test numerator closed form for linear confounder regularization."""
        alpha = _get_coef(sem, "X", "Z")
        beta = _get_coef(sem, "Y", "X")
        gamma = _get_coef(sem, "Y", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)
        latent_samples = sem.posterior.sample(3000)

        num_samples = sem._compute_causal_regularization_numerator_term_samples(
            observed=observed,
            treatment_name="X",
            outcome_name="Y",
            observed_name="X",
            latent_samples=latent_samples,
        )

        x = observed["X"]
        expected_num = (
            (beta + gamma * alpha / (1 + alpha**2)) * x**2 - gamma * alpha
        ) / (1 + alpha**2)

        assert torch.allclose(num_samples.mean(dim=1), expected_num, atol=0.35), (
            "Causal regularization numerator should match closed form "
            "((beta+gamma*alpha/(1+alpha^2))*x^2 - gamma*alpha)/(1+alpha^2)"
        )

    def test_causal_regularization_denominator_matches_closed_form(
        self, sem: SEM, values: dict[str, Tensor]
    ) -> None:
        """Test denominator closed form for linear confounder regularization."""
        alpha = _get_coef(sem, "X", "Z")

        observed: dict[str, Tensor] = {"X": values["X"]}
        sem.posterior.fit(observed)
        latent_samples = sem.posterior.sample(3000)

        denom_samples = sem._compute_causal_regularization_denominator_term_samples(
            observed=observed,
            treatment_name="X",
            observed_name="X",
            latent_samples=latent_samples,
        )

        x = observed["X"]
        expected_denom = x / (1 + alpha**2)

        assert torch.allclose(denom_samples.mean(dim=1), expected_denom, atol=0.2), (
            "Causal regularization denominator should match closed form x/(1+alpha^2)"
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
        samples = sem.posterior.sample(2000)

        expected_means: Tensor = (
            gamma * (observed["Y"] - beta * observed["X"]) + alpha * observed["X"]
        ) / (1 + alpha**2 + gamma**2)
        expected_vars: Tensor = Tensor([1 / (1 + alpha**2 + gamma**2)]).expand(10)

        mean_error: Tensor = torch.abs(expected_means - samples["Z"].mean(1))
        var_error: Tensor = torch.abs(expected_vars - samples["Z"].var(1))

        assert torch.all(mean_error < 0.08), (
            f"Posterior mean error too large: {mean_error}"
        )
        assert torch.all(var_error < 0.08), (
            f"Posterior variance error too large: {var_error}"
        )


class TestOvercontrolModel:
    """Tests for causal effect and causal bias in the overcontrol model.

    For the linear overcontrol model:
    - X = U_X
    - V = alpha * X + U_V
    - Y = beta * X + gamma * V + U_Y

    When observing X and V:
    - Expected causal effect: beta + gamma * alpha
    - Expected causal bias: -gamma * alpha
    """

    @pytest.fixture
    def overcontrol_sem(self) -> SEM:
        """Create a test SEM with overcontrol structure X -> V, X -> Y, V -> Y.

        Returns:
            A SEM with three linear variables and unit noise.
        """
        return SEM(
            variables=[
                LinearVariable(
                    name="X", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
                LinearVariable(
                    name="V",
                    parent_names=["X"],
                    sigma=1.0,
                    coefs={"X": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    name="Y",
                    parent_names=["X", "V"],
                    sigma=1.0,
                    coefs={"X": 2.0, "V": 3.0},
                    intercept=0.0,
                ),
            ]
        )

    @pytest.fixture
    def overcontrol_values(self, overcontrol_sem: SEM) -> dict[str, Tensor]:
        """Generate sample values from the overcontrol SEM.

        Args:
            overcontrol_sem: The structural equation model.

        Returns:
            Dictionary of generated values for all variables.
        """
        torch.manual_seed(42)
        return overcontrol_sem.generate(10)

    def test_causal_effect_overcontrol(
        self, overcontrol_sem: SEM, overcontrol_values: dict[str, Tensor]
    ) -> None:
        """Test that causal effect of X on Y equals beta + gamma * alpha when observing X and V.

        For the overcontrol model Y = beta*X + gamma*V + noise, where V = alpha*X + noise,
        the causal effect of X on Y is the total derivative dY/dX = beta + gamma * alpha.
        """
        alpha = _get_coef(overcontrol_sem, "V", "X")
        beta = _get_coef(overcontrol_sem, "Y", "X")
        gamma = _get_coef(overcontrol_sem, "Y", "V")

        observed: dict[str, Tensor] = {
            "X": overcontrol_values["X"],
            "V": overcontrol_values["V"],
        }

        overcontrol_sem.posterior.fit(observed)

        causal_effect = overcontrol_sem.causal_effect(
            observed, treatment_name="X", outcome_name="Y"
        )

        expected_causal_effect = torch.full_like(causal_effect, beta + gamma * alpha)

        assert torch.allclose(causal_effect, expected_causal_effect, atol=1e-5), (
            f"Causal effect should equal beta + gamma*alpha={beta + gamma * alpha}, got {causal_effect}"
        )

    def test_causal_bias_overcontrol(
        self, overcontrol_sem: SEM, overcontrol_values: dict[str, Tensor]
    ) -> None:
        """Test that causal bias of X on Y equals -gamma * alpha when observing X and V.

        For the overcontrol model, when we observe X and V, we overcontrol for V
        which is a mediator on the causal path from X to Y.
        The analytical causal bias is: -gamma * alpha
        """
        alpha = _get_coef(overcontrol_sem, "V", "X")
        gamma = _get_coef(overcontrol_sem, "Y", "V")

        observed: dict[str, Tensor] = {
            "X": overcontrol_values["X"],
            "V": overcontrol_values["V"],
        }

        overcontrol_sem.posterior.fit(observed)

        causal_bias = overcontrol_sem.causal_bias(
            observed, treatment_name="X", outcome_name="Y"
        )

        expected_causal_bias = torch.full_like(causal_bias, -gamma * alpha)

        assert torch.allclose(causal_bias, expected_causal_bias, atol=0.2), (
            f"Causal bias should equal -gamma*alpha={-gamma * alpha}, got {causal_bias}"
        )


class TestEndogenousSelectionModel:
    """Tests for causal effect and causal bias in the endogenous selection model.

    For the linear endogenous selection model:
    - X = U_X
    - Y = alpha * X + U_Y
    - V = beta * X + gamma * Y + U_V

    When observing X and V:
    - Expected causal effect: alpha
    - Expected causal bias: -(gamma * (beta + gamma * alpha)) / (1 + gamma^2)
    """

    @pytest.fixture
    def endogenous_selection_sem(self) -> SEM:
        """Create a test SEM with endogenous selection structure X -> Y, X -> V, Y -> V.

        Returns:
            A SEM with three linear variables and unit noise.
        """
        return SEM(
            variables=[
                LinearVariable(
                    name="X", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
                LinearVariable(
                    name="Y",
                    parent_names=["X"],
                    sigma=1.0,
                    coefs={"X": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    name="V",
                    parent_names=["X", "Y"],
                    sigma=1.0,
                    coefs={"X": 2.0, "Y": 3.0},
                    intercept=0.0,
                ),
            ]
        )

    @pytest.fixture
    def endogenous_selection_values(
        self, endogenous_selection_sem: SEM
    ) -> dict[str, Tensor]:
        """Generate sample values from the endogenous selection SEM.

        Args:
            endogenous_selection_sem: The structural equation model.

        Returns:
            Dictionary of generated values for all variables.
        """
        torch.manual_seed(42)
        return endogenous_selection_sem.generate(10)

    def test_causal_effect_endogenous_selection(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test that causal effect of X on Y equals alpha when observing X and V.

        For the endogenous selection model Y = alpha*X + noise,
        the causal effect of X on Y is the partial derivative dY/dX = alpha.
        """
        alpha = _get_coef(endogenous_selection_sem, "Y", "X")

        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"],
            "V": endogenous_selection_values["V"],
        }

        endogenous_selection_sem.posterior.fit(observed)

        causal_effect = endogenous_selection_sem.causal_effect(
            observed, treatment_name="X", outcome_name="Y"
        )

        expected_causal_effect = torch.full_like(causal_effect, alpha)

        assert torch.allclose(causal_effect, expected_causal_effect, atol=1e-5), (
            f"Causal effect should equal alpha={alpha}, got {causal_effect}"
        )

    def test_causal_bias_endogenous_selection(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test causal bias of X on Y when observing X and V in endogenous selection model.

        For the endogenous selection model, when we observe X and V,
        the analytical causal bias is: -(gamma * (beta + gamma * alpha)) / (1 + gamma^2)
        """
        alpha = _get_coef(endogenous_selection_sem, "Y", "X")
        beta = _get_coef(endogenous_selection_sem, "V", "X")
        gamma = _get_coef(endogenous_selection_sem, "V", "Y")

        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"],
            "V": endogenous_selection_values["V"],
        }

        endogenous_selection_sem.posterior.fit(observed)

        causal_bias = endogenous_selection_sem.causal_bias(
            observed, treatment_name="X", outcome_name="Y"
        )

        expected_causal_bias = torch.full_like(
            causal_bias, -(gamma * (beta + gamma * alpha)) / (1 + gamma**2)
        )

        assert torch.allclose(causal_bias, expected_causal_bias, atol=0.2), (
            f"Causal bias should equal -(gamma*(beta+gamma*alpha))/(1+gamma^2)={-(gamma * (beta + gamma * alpha)) / (1 + gamma**2)}, got {causal_bias}"
        )

    def test_contribution_observing_x_is_null(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test that the mean contribution when observing X is null.

        For the endogenous selection model X -> Y, X -> V, Y -> V,
        when we observe X, its mean contribution to causal bias should be zero
        because X is the treatment variable and there is no confounding path through X.

        The contribution from X is: -(du_fy + mid_term) * dx_diff / sigma_X
        - dx_diff = -1 (for treatment variable)
        - du_fy = 0 (dY/du_X = 0 because Y = alpha*X + U_Y depends on X, not u_X)
        - E[mid_term] = 0 (centered Y has zero correlation with u_X)
        """
        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"],
            "V": endogenous_selection_values["V"],
        }

        endogenous_selection_sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = endogenous_selection_sem.posterior.sample(2000)

        # Test du_fy = 0 for a single sample
        u_latent = {k: v[0, 0] for k, v in u_latent_samples.items()}
        obs = {k: v[0] for k, v in observed.items()}

        u_x = endogenous_selection_sem._get_u_observed(u_latent, obs, "X")
        f_mean_u = partial(
            endogenous_selection_sem._compute_outcome_mean_from_noise,
            latent=u_latent,
            observed=obs,
            input_name="X",
            treatment_name="X",
            outcome_name="Y",
        )
        du_fy = grad(f_mean_u)(u_x)

        assert torch.allclose(du_fy, torch.tensor(0.0), atol=1e-5), (
            f"du_fy should be 0 when observing X, got {du_fy}"
        )

        # Test mean of mid_term = 0 (centered Y has zero mean correlation with u_X)
        outcome_means = endogenous_selection_sem._compute_conditional_outcome_mean(
            u_latent_samples, observed, outcome_name="Y"
        )

        # Compute mid_term for each sample and observation, then average
        mid_terms = []
        num_obs = observed["X"].shape[0]
        num_samples = 2000

        for i in range(num_obs):
            sample_mid_terms = []
            for j in range(num_samples):
                u_latent = {k: v[i, j] for k, v in u_latent_samples.items()}
                obs = {k: v[i] for k, v in observed.items()}
                mid_term = endogenous_selection_sem._compute_bias_mid_term(
                    u_latent,
                    obs,
                    outcome_mean=outcome_means[i],
                    observed_name="X",
                    outcome_name="Y",
                )
                sample_mid_terms.append(mid_term)
            mid_terms.append(torch.stack(sample_mid_terms).mean())

        mean_mid_term = torch.stack(mid_terms).mean()

        assert torch.allclose(mean_mid_term, torch.tensor(0.0), atol=0.15), (
            f"mean mid_term should be 0 when observing X, got {mean_mid_term}"
        )

    def test_dx_diff_observing_v(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test that dx_diff = beta + gamma*alpha when observing V.

        For the endogenous selection model:
        - V = beta*X + gamma*Y + U_V
        - Y = alpha*X + U_Y

        The derivative dV/dX = beta + gamma * dY/dX = beta + gamma * alpha.
        """
        alpha = _get_coef(endogenous_selection_sem, "Y", "X")
        beta = _get_coef(endogenous_selection_sem, "V", "X")
        gamma = _get_coef(endogenous_selection_sem, "V", "Y")

        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"][:1],
            "V": endogenous_selection_values["V"][:1],
        }

        endogenous_selection_sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = endogenous_selection_sem.posterior.sample(1)
        u_latent = {k: v[0, 0] for k, v in u_latent_samples.items()}
        obs = {k: v[0] for k, v in observed.items()}

        f_mean_x = partial(
            endogenous_selection_sem._compute_target_mean,
            latent=u_latent,
            observed=obs,
            treatment_name="X",
            target_name="V",
        )
        dx_diff = grad(f_mean_x)(obs["X"])

        expected_dx_diff = beta + gamma * alpha

        assert torch.allclose(dx_diff, torch.tensor(expected_dx_diff), atol=1e-5), (
            f"dx_diff should equal beta + gamma*alpha = {expected_dx_diff}, got {dx_diff}"
        )

    def test_du_fy_observing_v_is_zero(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test that du_fy = 0 when observing V.

        For the endogenous selection model, the derivative dY/du_V should be 0
        because Y = alpha*X + U_Y does not depend on V or u_V.
        """
        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"][:1],
            "V": endogenous_selection_values["V"][:1],
        }

        endogenous_selection_sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = endogenous_selection_sem.posterior.sample(1)
        u_latent = {k: v[0, 0] for k, v in u_latent_samples.items()}
        obs = {k: v[0] for k, v in observed.items()}

        u_v = endogenous_selection_sem._get_u_observed(u_latent, obs, "V")

        f_mean_u = partial(
            endogenous_selection_sem._compute_outcome_mean_from_noise,
            latent=u_latent,
            observed=obs,
            input_name="V",
            treatment_name="X",
            outcome_name="Y",
        )
        du_fy = grad(f_mean_u)(u_v)

        assert torch.allclose(du_fy, torch.tensor(0.0), atol=1e-5), (
            f"du_fy should be 0 when observing V, got {du_fy}"
        )

    def test_mid_term_observing_v(
        self,
        endogenous_selection_sem: SEM,
        endogenous_selection_values: dict[str, Tensor],
    ) -> None:
        """Test that mean of mid_term = gamma/(1+gamma^2) when observing V.

        For the endogenous selection model, the mid_term captures the collider bias
        contribution through V. Its expected value should equal gamma / (1 + gamma^2).

        The mid_term formula is: -(Y_sample - Y_mean) * u_V
        where u_V = V - f_mean_V is the normalized residual of V.

        For the linear case, E[mid_term] = gamma * Var[u_Y | X, V] = gamma / (1 + gamma^2).
        """
        gamma = _get_coef(endogenous_selection_sem, "V", "Y")

        observed: dict[str, Tensor] = {
            "X": endogenous_selection_values["X"],
            "V": endogenous_selection_values["V"],
        }

        endogenous_selection_sem.posterior.fit(observed)

        torch.manual_seed(0)
        u_latent_samples = endogenous_selection_sem.posterior.sample(2000)
        outcome_means = endogenous_selection_sem._compute_conditional_outcome_mean(
            u_latent_samples, observed, outcome_name="Y"
        )

        # Compute mid_term for each sample and observation, then average
        mid_terms = []
        num_obs = observed["X"].shape[0]
        num_samples = 2000

        for i in range(num_obs):
            sample_mid_terms = []
            for j in range(num_samples):
                u_latent = {k: v[i, j] for k, v in u_latent_samples.items()}
                obs = {k: v[i] for k, v in observed.items()}
                mid_term = endogenous_selection_sem._compute_bias_mid_term(
                    u_latent,
                    obs,
                    outcome_mean=outcome_means[i],
                    observed_name="V",
                    outcome_name="Y",
                )
                sample_mid_terms.append(mid_term)
            mid_terms.append(torch.stack(sample_mid_terms).mean())

        mean_mid_term = torch.stack(mid_terms).mean()
        expected_mid_term = gamma / (1 + gamma**2)

        assert torch.allclose(
            mean_mid_term, torch.tensor(expected_mid_term), atol=0.15
        ), (
            f"mid_term mean should equal gamma/(1+gamma^2)={expected_mid_term}, got {mean_mid_term}"
        )
