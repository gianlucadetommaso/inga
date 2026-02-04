"""Causal effect computation mixin for Structural Equation Models."""

import torch
from torch import Tensor, vmap, no_grad
from torch.func import grad
from functools import partial

from steindag.variable.base import Variable
from steindag.approx_posterior.laplace import LaplacePosterior


class CausalBiasMixin:
    _variables: dict[str, Variable]
    posterior: LaplacePosterior

    def causal_bias(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        return self._causal_bias(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).mean(1)

    def causal_bias_var(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        return self._causal_bias(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).var(1)

    @no_grad()
    def _causal_bias(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        if outcome_name in observed:
            raise ValueError("`outcome_name` cannot be included in `observed_name`.")

        if treatment_name not in observed:
            raise ValueError("`treatment_name` must be included in `observed_name`.")

        if treatment_name == outcome_name:
            raise ValueError("`treatment_name` and `observed_name` cannot be equal.")

        u_latent_samples = self.posterior.sample(num_samples)
        outcome_means = self._cond_mean_outcome(
            u_latent_samples, observed, outcome_name=outcome_name
        )
        causal_bias = 0.0

        for observed_name in observed:

            @partial(vmap, in_dims=1, out_dims=0)
            def causal_bias_contrib_sample(
                u_latent_sample: dict[str, Tensor],
            ) -> Tensor:
                @vmap
                def causal_bias_contrib_sample_per_observation(
                    u_latent_per_observation: dict[str, Tensor],
                    observed_per_observation: dict[str, Tensor],
                    outcome_mean: Tensor,
                ) -> Tensor:
                    if observed_name == treatment_name:
                        dx_diff = -torch.ones_like(
                            observed_per_observation[treatment_name]
                        )
                    else:
                        fv_bar_x = partial(
                            self._fv_bar_x,
                            u_latent=u_latent_per_observation,
                            observed=observed_per_observation,
                            treatment_name=treatment_name,
                            target_name=observed_name,
                        )
                        dx_diff = grad(fv_bar_x)(
                            observed_per_observation[treatment_name]
                        )

                    fy_bar_u = partial(
                        self._fy_bar_u,
                        u_latent=u_latent_per_observation,
                        observed=observed_per_observation,
                        input_name=observed_name,
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                    )
                    inpt = self._get_u_observed(
                        u_latent_per_observation,
                        observed_per_observation,
                        observed_name,
                    )
                    du_fy = grad(fy_bar_u)(inpt)

                    mid_term = self._mid_term(
                        u_latent_per_observation,
                        observed_per_observation,
                        outcome_mean=outcome_mean,
                        observed_name=observed_name,
                        outcome_name=outcome_name,
                    )

                    return (
                        -(du_fy + mid_term)
                        * dx_diff
                        / self._variables[observed_name].sigma
                    )

                return causal_bias_contrib_sample_per_observation(
                    u_latent_sample, observed, outcome_means
                )

            causal_bias += causal_bias_contrib_sample(u_latent_samples).T

        return causal_bias

    def _fv_bar_x(
        self,
        treatment: Tensor,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        treatment_name: str,
        target_name: str,
    ) -> Tensor:
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name == target_name:
                return self._variables[name].f_bar(parents)
            if name == treatment_name:
                values[name] = treatment
            elif name in observed:
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Target variable '{target_name}' not found in the SEM.")

    def _fy_bar_u(
        self,
        inpt: Tensor,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        input_name: Tensor,
        treatment_name: str,
        outcome_name: str,
    ) -> Tensor:
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name == outcome_name:
                return self._variables[name].f_bar(parents)

            if name == treatment_name:
                values[name] = observed[treatment_name]
            elif name == input_name:
                values[name] = variable.f(parents, inpt)
            elif name in u_latent:
                values[name] = variable.f(parents, u_latent[name])
            else:
                f_bar = variable.f_bar(parents)
                values[name] = variable.f(
                    parents,
                    ((observed[name] - f_bar) / variable.sigma).detach(),
                    f_bar=f_bar,
                )

        raise ValueError(f"Outcome variable '{outcome_name}' not found in the SEM.")

    def _fv_bar_x(
        self,
        treatment: Tensor,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        treatment_name: str,
        target_name: str,
    ) -> Tensor:
        values: dict[str, Tensor] = {}
        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name == target_name:
                return self._variables[name].f_bar(parents)
            if name == treatment_name:
                values[name] = treatment
            elif name in observed:
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Target variable '{target_name}' not found in the SEM.")

    def _get_u_observed(
        self,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        observed_name: str,
    ) -> Tensor:
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                pa_name: parent
                for pa_name, parent in values.items()
                if pa_name in variable.parent_names
            }
            f_bar = variable.f_bar(parents)

            if name in observed:
                if name == observed_name:
                    return (observed[name] - f_bar) / variable.sigma
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name], f_bar)

        raise ValueError(f"Observed variable '{observed_name}' not found in the SEM.")

    def _cond_mean_outcome(
        self,
        u_latent_samples: dict[str, Tensor],
        observed: dict[str, Tensor],
        outcome_name: str,
    ) -> dict[str, Tensor]:
        @partial(vmap, in_dims=1, out_dims=0)
        def cond_mean_outcome_per_sample(u_latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def cond_mean_outcome_per_observation(
                u_latent_per_observation: dict[str, Tensor],
                observed_per_observation: dict[str, Tensor],
            ) -> Tensor:
                values: dict[str, Tensor] = {}

                for name, variable in self._variables.items():
                    parents = {
                        parent_name: values[parent_name]
                        for parent_name in variable.parent_names
                    }

                    if name == outcome_name:
                        return self._variables[name].f(
                            parents, u_latent_per_observation[name]
                        )

                    if name in observed_per_observation:
                        values[name] = observed_per_observation[name]
                    else:
                        values[name] = variable.f(
                            parents, u_latent_per_observation[name]
                        )

                raise ValueError(
                    f"Outcome variable '{outcome_name}' not found in the SEM."
                )

            return cond_mean_outcome_per_observation(u_latent_sample, observed)

        return cond_mean_outcome_per_sample(u_latent_samples).T.mean(1)

    def _mid_term(
        self,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        outcome_mean: Tensor,
        observed_name: str,
        outcome_name: str,
    ) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}
        u_observed: Tensor | None = None
        diff_term: Tensor | None = None

        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name in observed:
                f_bar = variable.f_bar(parents)
                if name == observed_name:
                    u_observed = (observed[name] - f_bar) / variable.sigma
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

            if name == outcome_name:
                diff_term = variable.f(parents, u_latent[name]) - outcome_mean

            if u_observed is not None and diff_term is not None:
                return -diff_term * u_observed

        raise ValueError(
            f"Outcome variable '{outcome_name}' or observed variable '{observed_name}' not found in the SEM."
        )
