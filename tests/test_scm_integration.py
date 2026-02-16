"""Integration tests for SCM workflows."""

import torch

from inga.scm.random import RandomSCMConfig, random_scm


class TestRandomSEMIntegration:
    """Integration tests for random SCM with posterior and causal queries."""

    def test_random_scm_posterior_and_causal_queries(self) -> None:
        """Generate a random DAG, fit posterior, and compute causal metrics."""
        config = RandomSCMConfig(
            num_variables=5,
            parent_prob=0.6,
            nonlinear_prob=0.5,
            seed=123,
        )
        scm = random_scm(config)

        torch.manual_seed(0)
        values = scm.generate(50)

        variable_names = list(scm._variables.keys())
        treatment_name = variable_names[1]
        outcome_name = variable_names[-1]
        observed = {treatment_name: values[treatment_name]}

        scm.posterior.fit(observed)
        causal_effect = scm.causal_effect(
            observed, treatment_name=treatment_name, outcome_name=outcome_name
        )
        causal_bias = scm.causal_bias(
            observed, treatment_name=treatment_name, outcome_name=outcome_name
        )

        assert causal_effect.shape == values[treatment_name].shape
        assert causal_bias.shape == values[treatment_name].shape
