"""Causal effect computation mixin for Structural Equation Models."""

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

        @partial(vmap, in_dims=1, out_dims=1)
        def f_bar_outcome_per_sample(u_latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def f_bar_outcome_per_observation(
                u_latent_per_observation: dict[str, Tensor],
                single_observed: dict[str, Tensor],
            ) -> Tensor:
                f_bar_outcome = partial(
                    self._f_bar_outcome,
                    u_latent=u_latent_per_observation,
                    observed={
                        name: values
                        for name, values in single_observed.items()
                        if name != treatment_name
                    },
                    treatment_name=treatment_name,
                    outcome_name=outcome_name,
                )

                return grad(f_bar_outcome)(single_observed[treatment_name])

            return f_bar_outcome_per_observation(u_latent_sample, observed)

        return f_bar_outcome_per_sample(u_latent_samples)

    def _f_bar_outcome(
        self,
        treatment: Tensor,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
    ) -> Tensor:
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = {
                parent_name: values[parent_name]
                for parent_name in variable.parent_names
            }

            if name == treatment_name:
                values[name] = treatment
            elif name == outcome_name:
                return self._variables[name].f_bar(parents)
            elif name in observed:
                values[name] = observed[name]
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Outcome variable '{outcome_name}' not found in the SEM.")
