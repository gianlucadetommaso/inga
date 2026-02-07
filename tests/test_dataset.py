"""Tests for SEM dataset generation."""

import torch
import pytest

from pathlib import Path

from steindag.sem.dataset import (
    SEMDatasetConfig,
    generate_sem_dataset,
    load_sem_dataset,
)
from steindag.sem.random import RandomSEMConfig


class TestSEMDataset:
    """Unit tests for SEM dataset generation."""

    def test_dataset_shapes_and_query_rules(self) -> None:
        """Ensure dataset matches schema and query constraints."""
        config = SEMDatasetConfig(
            sem_config=RandomSEMConfig(num_variables=5, parent_prob=0.5, seed=1),
            num_samples=200,
            num_queries=5,
            min_observed=1,
            seed=123,
        )
        dataset = generate_sem_dataset(config)

        assert len(dataset.data) == 5
        for values in dataset.data.values():
            assert values.shape == (config.num_samples,)

        assert len(dataset.queries) == config.num_queries
        for query in dataset.queries:
            assert query.treatment_name in query.observed_names
            assert query.outcome_name not in query.observed_names

        assert len(dataset.causal_effects) == config.num_queries
        assert len(dataset.causal_biases) == config.num_queries
        assert len(dataset.causal_regularizations) == config.num_queries

        for key, effect in dataset.causal_effects.items():
            assert effect.shape == (config.num_samples,)
            assert key in dataset.causal_biases
            assert key in dataset.causal_regularizations

    def test_invalid_dataset_config(self) -> None:
        """Ensure invalid configs raise errors."""
        with pytest.raises(ValueError, match="num_samples"):
            generate_sem_dataset(
                SEMDatasetConfig(
                    sem_config=RandomSEMConfig(num_variables=3),
                    num_samples=0,
                )
            )

        with pytest.raises(ValueError, match="num_queries"):
            generate_sem_dataset(
                SEMDatasetConfig(
                    sem_config=RandomSEMConfig(num_variables=3),
                    num_queries=0,
                )
            )

    def test_dataset_save_load_round_trip(self, tmp_path: Path) -> None:
        """Ensure datasets can be saved and loaded with SEM spec."""
        config = SEMDatasetConfig(
            sem_config=RandomSEMConfig(num_variables=4, parent_prob=0.5, seed=2),
            num_samples=100,
            num_queries=3,
            min_observed=1,
            seed=10,
        )
        dataset = generate_sem_dataset(config)
        target = tmp_path / "sem_dataset"
        dataset.save(target)

        loaded = load_sem_dataset(target)
        assert loaded.data.keys() == dataset.data.keys()
        for name in dataset.data:
            assert torch.allclose(loaded.data[name], dataset.data[name])

        assert len(loaded.queries) == len(dataset.queries)
        for original, restored in zip(dataset.queries, loaded.queries):
            assert original == restored

        for key, effect in dataset.causal_effects.items():
            assert torch.allclose(effect, loaded.causal_effects[key])
        for key, bias in dataset.causal_biases.items():
            assert torch.allclose(bias, loaded.causal_biases[key])
        for key, regularization in dataset.causal_regularizations.items():
            assert torch.allclose(
                regularization, loaded.causal_regularizations[key]
            )

    def test_dataset_single_seed_drives_sem_and_queries(self) -> None:
        """Ensure one dataset seed is enough for deterministic SEM + queries.

        If sem_config.seed differs but dataset seed is equal, generation should
        still be deterministic and identical because dataset seed takes priority.
        """
        config_a = SEMDatasetConfig(
            sem_config=RandomSEMConfig(num_variables=5, parent_prob=0.5, seed=11),
            num_samples=120,
            num_queries=3,
            min_observed=1,
            seed=1234,
        )
        config_b = SEMDatasetConfig(
            sem_config=RandomSEMConfig(num_variables=5, parent_prob=0.5, seed=999),
            num_samples=120,
            num_queries=3,
            min_observed=1,
            seed=1234,
        )

        ds_a = generate_sem_dataset(config_a)
        ds_b = generate_sem_dataset(config_b)

        assert ds_a.queries == ds_b.queries
        assert list(ds_a.sem._variables.keys()) == list(ds_b.sem._variables.keys())

        for name in ds_a.data:
            assert torch.allclose(ds_a.data[name], ds_b.data[name])

        for key in ds_a.causal_effects:
            assert torch.allclose(ds_a.causal_effects[key], ds_b.causal_effects[key])
            assert torch.allclose(ds_a.causal_biases[key], ds_b.causal_biases[key])
            assert torch.allclose(
                ds_a.causal_regularizations[key], ds_b.causal_regularizations[key]
            )
