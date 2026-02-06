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

        for key, effect in dataset.causal_effects.items():
            assert effect.shape == (config.num_samples,)
            assert key in dataset.causal_biases

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