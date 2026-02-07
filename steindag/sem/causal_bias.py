"""Causal bias computation mixin for Structural Equation Models."""

from __future__ import annotations

import torch
from torch import Tensor, vmap, no_grad
from torch.func import grad
from functools import partial
from typing import TYPE_CHECKING

from steindag.sem.causal_effect import CausalEffectMixin

if TYPE_CHECKING:
    from steindag.variable.base import Variable
    from steindag.approx_posterior.laplace import LaplacePosterior


class CausalBiasMixin(CausalEffectMixin):
    """Mixin class providing causal bias computation methods.

    The causal bias captures the discrepancy between the observational effect
    and the interventional (causal) effect, induced by confounding, selection,
    or overcontrol.

    This mixin assumes the class also provides the attributes required by
    CausalEffectMixin.
    """

    _variables: "dict[str, Variable]"
    posterior: "LaplacePosterior"

    def causal_bias(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute the expected causal bias for each observation.

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw.

        Returns:
            Tensor of shape (num_observations,) containing the mean causal bias.
        """
        return self._compute_causal_bias_samples(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).mean(dim=1)

    def causal_bias_var(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute posterior variance of the causal bias for each observation.

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw.

        Returns:
            Tensor of shape (num_observations,) containing bias variance.
        """
        return self._compute_causal_bias_samples(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).var(dim=1)

    def causal_regularization(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute the expected causal regularization term.

        This averages causal regularization samples over the posterior-sample
        dimension (dimension 1).

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw.

        Returns:
            Tensor of shape (num_observations,) with causal regularization values.
        """
        return self._compute_causal_regularization_samples(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).mean(dim=1)

    @no_grad()
    def _compute_causal_bias_samples(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute causal bias samples for each observation.

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw.

        Returns:
            Tensor of shape (num_observations, num_samples) containing causal
            bias samples for each observation.
        """
        self._validate_causal_query(observed, treatment_name, outcome_name)

        latent_samples = self.posterior.sample(num_samples)
        outcome_means = self._compute_conditional_outcome_mean(
            latent_samples, observed, outcome_name
        )

        bias_samples = torch.zeros(
            observed[treatment_name].shape[0], num_samples, device=outcome_means.device
        )

        for observed_name in observed:

            @partial(vmap, in_dims=1, out_dims=0)
            def bias_contrib_per_sample(
                latent_sample: dict[str, Tensor],
            ) -> Tensor:
                @vmap
                def bias_contrib_per_observation(
                    latent_per_obs: dict[str, Tensor],
                    observed_per_obs: dict[str, Tensor],
                    outcome_mean: Tensor,
                ) -> Tensor:
                    dx_diff = self._compute_dx_diff(
                        observed_name=observed_name,
                        treatment_name=treatment_name,
                        latent=latent_per_obs,
                        observed=observed_per_obs,
                    )

                    du_fy = self._compute_du_fy(
                        observed_name=observed_name,
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                        latent=latent_per_obs,
                        observed=observed_per_obs,
                    )

                    mid_term = self._compute_bias_mid_term(
                        latent=latent_per_obs,
                        observed=observed_per_obs,
                        outcome_mean=outcome_mean,
                        observed_name=observed_name,
                        outcome_name=outcome_name,
                    )

                    sigma = self._variables[observed_name].sigma
                    return -(du_fy + mid_term) * dx_diff / sigma

                return bias_contrib_per_observation(
                    latent_sample, observed, outcome_means
                )

            bias_samples += bias_contrib_per_sample(latent_samples).T

        return bias_samples
    
    @no_grad()
    def _compute_causal_regularization_samples(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute causal regularization samples for each observation.

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw.

        Returns:
            Tensor of shape (num_observations, num_samples) containing causal
            regularization samples.
        """
        self._validate_causal_query(observed, treatment_name, outcome_name)

        latent_samples = self.posterior.sample(num_samples)

        regularization_samples = torch.zeros(
            observed[treatment_name].shape[0],
            num_samples,
            device=observed[treatment_name].device,
        )

        for observed_name in observed:
            num = self._compute_causal_regularization_numerator_term_samples(
                observed,
                treatment_name,
                outcome_name,
                observed_name,
                latent_samples,
            )
            denom = self._compute_causal_regularization_denominator_term_samples(
                observed,
                treatment_name,
                observed_name,
                latent_samples,
            )
            regularization_samples += num / (denom + 1e-6)

        return regularization_samples

    def _compute_dx_diff(
        self,
        observed_name: str,
        treatment_name: str,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
    ) -> Tensor:
        """Compute derivative of observed variable with respect to treatment."""
        if observed_name == treatment_name:
            return -torch.ones_like(observed[treatment_name])

        f_mean_x = partial(
            self._compute_target_mean,
            latent=latent,
            observed=observed,
            treatment_name=treatment_name,
            target_name=observed_name,
        )
        return grad(f_mean_x)(observed[treatment_name])

    def _compute_du_fy(
        self,
        observed_name: str,
        treatment_name: str,
        outcome_name: str,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
    ) -> Tensor:
        """Compute derivative of outcome mean w.r.t. observed variable noise."""
        f_mean_u = partial(
            self._compute_outcome_mean_from_noise,
            latent=latent,
            observed=observed,
            input_name=observed_name,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
        )
        noise = self._get_u_observed(latent, observed, observed_name)
        return grad(f_mean_u)(noise)

    def _compute_target_mean(
        self,
        treatment: Tensor,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        treatment_name: str,
        target_name: str,
    ) -> Tensor:
        """Compute the mean function for a target variable under treatment."""
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name == target_name:
                return variable.f_mean(parents)
            if name == treatment_name:
                values[name] = treatment
            elif name in observed:
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, latent[name])

        raise ValueError(f"Target variable '{target_name}' not found in the SEM.")

    def _get_u_observed(
        self,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        observed_name: str,
    ) -> Tensor:
        """Recover the normalized noise for an observed variable."""
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)
            f_mean = variable.f_mean(parents)

            if name in observed:
                if name == observed_name:
                    return (observed[name] - f_mean) / variable.sigma
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, latent[name], f_mean)

        raise ValueError(f"Observed variable '{observed_name}' not found in the SEM.")
    
    @no_grad()
    def _compute_causal_regularization_numerator_term_samples(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        observed_name: str,
        latent_samples: dict[str, Tensor],
    ) -> Tensor:
        """Compute numerator term samples for causal regularization.

        The term is computed per observation/per posterior sample as:
            -(du_fy + mid_term) * dx_diff
        where the regularization mid-term uses the non-centered outcome value.

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            observed_name: Name of the observed variable whose contribution is
                being computed.
            latent_samples: Posterior latent samples with shape
                (num_observations, num_samples) per variable.

        Returns:
            Tensor of shape (num_observations, num_samples) containing numerator
            term samples.
        """

        @partial(vmap, in_dims=1, out_dims=0)
        def numerator_term_per_sample(latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def numerator_term_per_observation(
                latent_per_obs: dict[str, Tensor],
                observed_per_obs: dict[str, Tensor],
            ) -> Tensor:
                dx_diff = self._compute_dx_diff(
                    observed_name=observed_name,
                    treatment_name=treatment_name,
                    latent=latent_per_obs,
                    observed=observed_per_obs,
                )

                du_fy = self._compute_du_fy(
                    observed_name=observed_name,
                    treatment_name=treatment_name,
                    outcome_name=outcome_name,
                    latent=latent_per_obs,
                    observed=observed_per_obs,
                )

                mid_term = self._compute_regularizer_mid_term(
                    latent=latent_per_obs,
                    observed=observed_per_obs,
                    outcome_name=outcome_name,
                    observed_name=observed_name,
                )

                return -(du_fy + mid_term) * dx_diff

            return numerator_term_per_observation(latent_sample, observed)

        return numerator_term_per_sample(latent_samples).T

    @no_grad()
    def _compute_causal_regularization_denominator_term_samples(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        observed_name: str,
        latent_samples: dict[str, Tensor],
    ) -> Tensor:
        """Compute denominator term samples for causal regularization.

        The denominator term is computed per observation/per sample as:
            u_observed * dx_diff

        Args:
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            observed_name: Name of the observed variable whose contribution is
                being computed.
            latent_samples: Posterior latent samples with shape
                (num_observations, num_samples) per variable.

        Returns:
            Tensor of shape (num_observations, num_samples) containing denominator
            term samples.
        """

        @partial(vmap, in_dims=1, out_dims=0)
        def denominator_term_per_sample(latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def denominator_term_per_observation(
                latent_per_obs: dict[str, Tensor],
                observed_per_obs: dict[str, Tensor],
            ) -> Tensor:
                dx_diff = self._compute_dx_diff(
                    observed_name=observed_name,
                    treatment_name=treatment_name,
                    latent=latent_per_obs,
                    observed=observed_per_obs,
                )

                u_observed = self._compute_u_observed(
                    latent=latent_per_obs,
                    observed=observed_per_obs,
                    observed_name=observed_name,
                )

                return u_observed * dx_diff

            return denominator_term_per_observation(latent_sample, observed)

        return denominator_term_per_sample(latent_samples).T

    def _compute_conditional_outcome_mean(
        self,
        latent_samples: dict[str, Tensor],
        observed: dict[str, Tensor],
        outcome_name: str,
    ) -> Tensor:
        """Compute conditional mean of the outcome for each observation."""

        @partial(vmap, in_dims=1, out_dims=0)
        def outcome_mean_per_sample(latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def outcome_mean_per_observation(
                latent_per_obs: dict[str, Tensor],
                observed_per_obs: dict[str, Tensor],
            ) -> Tensor:
                values: dict[str, Tensor] = {}

                for name, variable in self._variables.items():
                    parents = self._get_parent_values(variable, values)

                    if name == outcome_name:
                        return variable.f(parents, latent_per_obs[name])

                    if name in observed_per_obs:
                        values[name] = observed_per_obs[name]
                    else:
                        values[name] = variable.f(parents, latent_per_obs[name])

                raise ValueError(
                    f"Outcome variable '{outcome_name}' not found in the SEM."
                )

            return outcome_mean_per_observation(latent_sample, observed)

        return outcome_mean_per_sample(latent_samples).T.mean(dim=1)

    def _compute_bias_mid_term(
        self,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        outcome_mean: Tensor,
        observed_name: str,
        outcome_name: str,
    ) -> Tensor:
        """Compute the mid-term component for a given observed variable."""
        values: dict[str, Tensor] = {}
        u_observed: Tensor | None = None
        diff_term: Tensor | None = None

        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name in observed:
                f_mean = variable.f_mean(parents)
                if name == observed_name:
                    u_observed = (observed[name] - f_mean) / variable.sigma
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, latent[name])

            if name == outcome_name:
                diff_term = variable.f(parents, latent[name]) - outcome_mean

            if u_observed is not None and diff_term is not None:
                return -diff_term * u_observed

        raise ValueError(
            f"Outcome variable '{outcome_name}' or observed variable '{observed_name}' not found in the SEM."
        )
    
    def _compute_u_observed(
        self,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        observed_name: str,
    ) -> Tensor:
        """Recover normalized noise for a specific observed variable.

        Args:
            latent: Dictionary mapping latent variable names to noise values.
            observed: Dictionary mapping observed variable names to values.
            observed_name: Name of the observed variable whose normalized noise
                is requested.

        Returns:
            Normalized noise value for the requested observed variable.

        Raises:
            ValueError: If observed_name is not found in the SEM.
        """
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name in observed:
                f_mean = variable.f_mean(parents)
                if name == observed_name:
                    return (observed[name] - f_mean) / variable.sigma
                
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, latent[name])

        raise ValueError(
            f"Observed variable '{observed_name}' not found in the SEM."
        )
    

    def _compute_regularizer_mid_term(
        self,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        outcome_name: str,
        observed_name: str,
    ) -> Tensor:
        """Compute the regularization-specific mid-term contribution.

        Unlike the causal-bias mid-term, this term uses the non-centered outcome
        value and is defined as:
            - outcome_value * u_observed

        Args:
            latent: Dictionary mapping latent variable names to noise values.
            observed: Dictionary mapping observed variable names to values.
            outcome_name: Name of the outcome variable.
            observed_name: Name of the observed variable used to compute
                the normalized noise term.

        Returns:
            Regularization mid-term value.

        Raises:
            ValueError: If outcome_name or observed_name are not found.
        """
        values: dict[str, Tensor] = {}
        u_observed: Tensor | None = None
        outcome_values: Tensor | None = None

        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name in observed:
                f_mean = variable.f_mean(parents)
                if name == observed_name:
                    u_observed = (observed[name] - f_mean) / variable.sigma
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, latent[name])

            if name == outcome_name:
                outcome_values = values[name]

            if u_observed is not None and outcome_values is not None:
                return -outcome_values * u_observed

        raise ValueError(
            f"Outcome variable '{outcome_name}' or observed variable '{observed_name}' not found in the SEM."
        )

    def _compute_outcome_mean_from_noise(
        self,
        noise: Tensor,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        input_name: str,
        treatment_name: str,
        outcome_name: str,
    ) -> Tensor:
        """Compute outcome mean when injecting noise for a specific observed variable."""
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name == outcome_name:
                return variable.f_mean(parents)

            if name == treatment_name:
                values[name] = observed[treatment_name]
            elif name == input_name:
                values[name] = variable.f(parents, noise)
            elif name in latent:
                values[name] = variable.f(parents, latent[name])
            else:
                f_mean = variable.f_mean(parents)
                residual = ((observed[name] - f_mean) / variable.sigma).detach()
                values[name] = variable.f(parents, residual, f_mean=f_mean)

        raise ValueError(f"Outcome variable '{outcome_name}' not found in the SEM.")
