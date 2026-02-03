"""Causal effect computation mixin for Structural Equation Models."""

from torch import Tensor, vmap, no_grad
from torch.func import grad
from functools import partial

from steindag.variable.base import Variable
from steindag.approx_posterior.laplace import LaplacePosterior


class CausalEffectMixin:
    """Mixin class providing causal effect computation methods.

    This mixin requires the class to have:
        - _variables: dict[str, Variable] - mapping of variable names to Variable objects
        - posterior: LaplacePosterior - posterior inference object with sample() method
    """

    _variables: dict[str, Variable]
    posterior: LaplacePosterior

    def causal_effect(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute the causal effect of treatment on outcome.

        The causal effect is the expected value of the gradient of the outcome's
        mean function with respect to the treatment, computed over posterior
        samples of the latent variables. For linear models, this equals the
        direct causal coefficient from treatment to outcome.

        Args:
            observed: Dictionary mapping observed variable names to their tensor values.
                Must include treatment_name but not outcome_name.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw. Defaults to 1000.

        Returns:
            Tensor of shape (num_observations,) containing the causal effect
            estimate for each observation.

        Raises:
            ValueError: If outcome_name is in observed, treatment_name is not in
                observed, or treatment_name equals outcome_name.
        """
        return self._causal_effect(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).mean(1)

    def causal_effect_var(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute the variance of the causal effect estimate.

        Computes the variance of the causal effect over posterior samples.
        For linear models, this variance is zero since the causal effect
        is constant.

        Args:
            observed: Dictionary mapping observed variable names to their tensor values.
                Must include treatment_name but not outcome_name.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw. Defaults to 1000.

        Returns:
            Tensor of shape (num_observations,) containing the causal effect
            variance for each observation.

        Raises:
            ValueError: If outcome_name is in observed, treatment_name is not in
                observed, or treatment_name equals outcome_name.
        """
        return self._causal_effect(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).var(1)

    @no_grad()
    def _causal_effect(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
        num_samples: int = 1000,
    ) -> Tensor:
        """Compute causal effect samples for each observation.

        Computes the causal effect of treatment on outcome by taking the gradient
        of the outcome's mean function with respect to the treatment, over
        posterior samples of latent variables.

        Args:
            observed: Dictionary mapping observed variable names to their tensor values.
                Must include treatment_name but not outcome_name.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.
            num_samples: Number of posterior samples to draw. Defaults to 1000.

        Returns:
            Tensor of shape (num_observations, num_samples) containing causal
            effect samples for each observation.

        Raises:
            ValueError: If outcome_name is in observed, treatment_name is not in
                observed, or treatment_name equals outcome_name.
        """
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
        """Compute the mean function of the outcome variable.

        Args:
            treatment: Value of the treatment variable.
            u_latent: Dictionary mapping latent variable names to noise values.
            observed: Dictionary mapping observed variable names to their values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.

        Returns:
            The mean function value of the outcome variable.

        Raises:
            ValueError: If outcome variable is not found in the SEM.
        """
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
