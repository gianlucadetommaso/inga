"""Integration tests for SEM workflows."""

import torch

from steindag.sem.random import RandomSEMConfig, random_sem


class TestRandomSEMIntegration:
    """Integration tests for random SEM with posterior and causal queries."""

    def test_random_sem_posterior_and_causal_queries(self) -> None:
        """Generate a random DAG, fit posterior, and compute causal metrics."""
        config = RandomSEMConfig(
            num_variables=5,
            parent_prob=0.6,
            nonlinear_prob=0.5,
            seed=123,
        )
        sem = random_sem(config)

        torch.manual_seed(0)
        values = sem.generate(50)

        variable_names = list(sem._variables.keys())
        treatment_name = variable_names[1]
        outcome_name = variable_names[-1]
        observed = {treatment_name: values[treatment_name]}

        sem.posterior.fit(observed)
        causal_effect = sem.causal_effect(
            observed, treatment_name=treatment_name, outcome_name=outcome_name
        )
        causal_bias = sem.causal_bias(
            observed, treatment_name=treatment_name, outcome_name=outcome_name
        )

        assert causal_effect.shape == values[treatment_name].shape
        assert causal_bias.shape == values[treatment_name].shape