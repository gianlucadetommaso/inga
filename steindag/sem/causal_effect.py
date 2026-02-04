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
        mediator_names = {
            name
            for name in observed
            if self._is_on_causal_path(name, treatment_name, outcome_name)
        }

        @partial(vmap, in_dims=1, out_dims=0)
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
                        if name != treatment_name and name not in mediator_names
                    },
                    mediator_observed={
                        name: values
                        for name, values in single_observed.items()
                        if name in mediator_names
                    },
                    treatment_name=treatment_name,
                    outcome_name=outcome_name,
                )

                return grad(f_bar_outcome)(single_observed[treatment_name])

            return f_bar_outcome_per_observation(u_latent_sample, observed)

        return f_bar_outcome_per_sample(u_latent_samples).T

    def _f_bar_outcome(
        self,
        treatment: Tensor,
        u_latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        mediator_observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
    ) -> Tensor:
        """Compute the mean function of the outcome variable.

        For mediators (observed variables on the causal path), computes the
        counterfactual value under intervention by deriving the noise from
        observed values and recomputing with the intervention treatment.

        Args:
            treatment: Value of the treatment variable.
            u_latent: Dictionary mapping latent variable names to noise values.
            observed: Dictionary mapping observed variable names to their values
                (excluding treatment and mediators).
            mediator_observed: Dictionary mapping mediator names to their observed
                values. Used to compute counterfactual mediator values under
                intervention.
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
            elif name in mediator_observed:
                # Compute noise from observed mediator value (detached from gradient)
                # and use it to reconstruct the counterfactual mediator under intervention.
                # The noise is detached so the gradient flows only through f_bar.
                f_bar = variable.f_bar(parents)
                u = ((mediator_observed[name] - f_bar) / variable.sigma).detach()
                values[name] = variable.f(parents, u, f_bar=f_bar)
            else:
                values[name] = variable.f(parents, u_latent[name])

        raise ValueError(f"Outcome variable '{outcome_name}' not found in the SEM.")

    def _is_on_causal_path(
        self,
        variable_name: str,
        treatment_name: str,
        outcome_name: str,
    ) -> bool:
        """Check if a variable is on a causal path from treatment to outcome.

        A variable is on a causal path if it is a proper descendant of the
        treatment and an ancestor of the outcome (or the outcome itself).
        The treatment itself is not considered to be on the causal path.

        Args:
            variable_name: Name of the variable to check.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.

        Returns:
            True if the variable is on a causal path from treatment to outcome,
            False otherwise. Returns False if variable_name equals treatment_name.

        Raises:
            ValueError: If any of the variable names are not in the SEM.
        """
        for name in [variable_name, treatment_name, outcome_name]:
            if name not in self._variables:
                raise ValueError(f"Variable '{name}' not found in the SEM.")

        # Treatment itself is not on the causal path
        if variable_name == treatment_name:
            return False

        # A variable is on a causal path if:
        # 1. It is a descendant of the treatment
        # 2. The outcome is reachable from it (is outcome or ancestor of outcome)

        descendants_of_treatment = self._get_descendants(treatment_name)

        ancestors_of_outcome = self._get_ancestors(outcome_name)
        ancestors_of_outcome.add(outcome_name)

        return (
            variable_name in descendants_of_treatment
            and variable_name in ancestors_of_outcome
        )

    def _get_descendants(self, variable_name: str) -> set[str]:
        """Get all descendants of a variable in the DAG.

        Args:
            variable_name: Name of the variable.

        Returns:
            Set of names of all descendant variables.
        """
        descendants: set[str] = set()
        to_visit = [variable_name]

        while to_visit:
            current = to_visit.pop()
            for name, variable in self._variables.items():
                if current in variable.parent_names and name not in descendants:
                    descendants.add(name)
                    to_visit.append(name)

        return descendants

    def _get_ancestors(self, variable_name: str) -> set[str]:
        """Get all ancestors of a variable in the DAG.

        Args:
            variable_name: Name of the variable.

        Returns:
            Set of names of all ancestor variables.
        """
        ancestors: set[str] = set()
        to_visit = list(self._variables[variable_name].parent_names)

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self._variables[current].parent_names)

        return ancestors
