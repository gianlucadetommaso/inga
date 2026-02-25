"""Causal effect computation mixin for Structural Causal Models."""

from torch import Tensor, vmap, no_grad
from torch.func import grad
from functools import partial
from typing import TYPE_CHECKING

from inga.scm.variable.gaussian import GaussianVariable
from inga.scm.variable.categorical import CategoricalVariable

if TYPE_CHECKING:
    from inga.scm.variable.base import Variable
    from inga.approx_posterior.laplace import LaplacePosterior


class CausalEffectMixin:
    """Mixin class providing causal effect computation methods.

    This mixin requires the class to have:
        - _variables: dict[str, Variable] - mapping of variable names to Variable objects
        - posterior: LaplacePosterior - posterior inference object with sample() method
    """

    _variables: "dict[str, Variable]"
    posterior: "LaplacePosterior"

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
        return self._compute_causal_effect_samples(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).mean(dim=1)

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
        return self._compute_causal_effect_samples(
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            num_samples=num_samples,
        ).var(dim=1)

    @no_grad()
    def _compute_causal_effect_samples(
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
        self._validate_causal_query(observed, treatment_name, outcome_name)

        latent_samples = self.posterior.sample(num_samples)
        mediator_names = self._find_mediators(observed, treatment_name, outcome_name)

        @partial(vmap, in_dims=1, out_dims=0)
        def compute_effect_per_sample(latent_sample: dict[str, Tensor]) -> Tensor:
            @vmap
            def compute_effect_per_observation(
                latent_per_obs: dict[str, Tensor],
                observed_per_obs: dict[str, Tensor],
            ) -> Tensor:
                outcome_f_mean = partial(
                    self._compute_outcome_mean_under_intervention,
                    latent=latent_per_obs,
                    observed=self._exclude_keys(
                        observed_per_obs, {treatment_name} | mediator_names
                    ),
                    mediator_observed=self._filter_keys(
                        observed_per_obs, mediator_names
                    ),
                    treatment_name=treatment_name,
                    outcome_name=outcome_name,
                )
                return grad(outcome_f_mean)(observed_per_obs[treatment_name])

            return compute_effect_per_observation(latent_sample, observed)

        return compute_effect_per_sample(latent_samples).T

    def _validate_causal_query(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
    ) -> None:
        """Validate inputs for causal effect/bias computation.

        Args:
            observed: Dictionary of observed variable values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.

        Raises:
            ValueError: If the query is invalid.
        """
        self._validate_supported_variables_for_causal_quantities()

        if outcome_name in observed:
            raise ValueError("`outcome_name` cannot be included in `observed_name`.")
        if treatment_name not in observed:
            raise ValueError("`treatment_name` must be included in `observed_name`.")
        if treatment_name == outcome_name:
            raise ValueError("`treatment_name` and `observed_name` cannot be equal.")

    def _validate_supported_variables_for_causal_quantities(self) -> None:
        """Ensure causal quantities are used with supported variable families."""
        unsupported = [
            name
            for name, variable in self._variables.items()
            if not isinstance(variable, (GaussianVariable, CategoricalVariable))
        ]
        if unsupported:
            raise ValueError(
                "Causal effect and causal bias are currently supported only for "
                "SCMs with GaussianVariable/CategoricalVariable nodes. "
                f"Found unsupported variables: {sorted(unsupported)}."
            )

    def _find_mediators(
        self,
        observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
    ) -> set[str]:
        """Find observed variables that are mediators on the causal path.

        Args:
            observed: Dictionary of observed variable values.
            treatment_name: Name of the treatment variable.
            outcome_name: Name of the outcome variable.

        Returns:
            Set of variable names that are mediators.
        """
        return {
            name
            for name in observed
            if self._is_on_causal_path(name, treatment_name, outcome_name)
        }

    def _compute_outcome_mean_under_intervention(
        self,
        treatment: Tensor,
        latent: dict[str, Tensor],
        observed: dict[str, Tensor],
        mediator_observed: dict[str, Tensor],
        treatment_name: str,
        outcome_name: str,
    ) -> Tensor:
        """Compute the mean function of the outcome variable under intervention.

        For mediators (observed variables on the causal path), computes the
        counterfactual value under intervention by deriving the noise from
        observed values and recomputing with the intervention treatment.

        Args:
            treatment: Value of the treatment variable.
            latent: Dictionary mapping latent variable names to noise values.
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
            ValueError: If outcome variable is not found in the SCM.
        """
        values: dict[str, Tensor] = {}

        for name, variable in self._variables.items():
            parents = self._get_parent_values(variable, values)

            if name == treatment_name:
                values[name] = treatment
            elif name == outcome_name:
                return variable.f_mean(parents)
            elif name in observed:
                values[name] = observed[name]
            elif name in mediator_observed:
                # Compute noise from observed mediator value (detached from gradient)
                # and use it to reconstruct the counterfactual mediator under intervention.
                # The noise is detached so the gradient flows only through f_mean.
                f_mean = variable.f_mean(parents)
                noise = variable.infer_noise(
                    parents=parents,
                    observed=mediator_observed[name],
                ).detach()
                values[name] = variable.f_from_mean(f_mean=f_mean, u=noise)
            else:
                values[name] = variable.f(parents, latent[name])

        raise ValueError(f"Outcome variable '{outcome_name}' not found in the SCM.")

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
            ValueError: If any of the variable names are not in the SCM.
        """
        for name in [variable_name, treatment_name, outcome_name]:
            if name not in self._variables:
                raise ValueError(f"Variable '{name}' not found in the SCM.")

        if variable_name == treatment_name:
            return False

        descendants_of_treatment = self._get_descendants(treatment_name)
        ancestors_of_outcome = self._get_ancestors(outcome_name) | {outcome_name}

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

    def _get_parent_values(
        self,
        variable: "Variable",
        values: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Extract parent values for a variable from computed values.

        Args:
            variable: The variable whose parents to extract.
            values: Dictionary of computed variable values.

        Returns:
            Dictionary mapping parent names to their values.
        """
        return {
            parent_name: values[parent_name] for parent_name in variable.parent_names
        }

    @staticmethod
    def _exclude_keys(d: dict[str, Tensor], keys: set[str]) -> dict[str, Tensor]:
        """Return a new dict excluding specified keys.

        Args:
            d: The source dictionary.
            keys: Keys to exclude.

        Returns:
            New dictionary without the specified keys.
        """
        return {k: v for k, v in d.items() if k not in keys}

    @staticmethod
    def _filter_keys(d: dict[str, Tensor], keys: set[str]) -> dict[str, Tensor]:
        """Return a new dict containing only specified keys.

        Args:
            d: The source dictionary.
            keys: Keys to include.

        Returns:
            New dictionary with only the specified keys.
        """
        return {k: v for k, v in d.items() if k in keys}
