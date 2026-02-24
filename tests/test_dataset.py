"""Tests for SCM dataset generation."""

import torch
import pytest

from pathlib import Path

from inga.scm.base import SCM
from inga.scm.dataset import (
    SCMDataset,
    SCMDatasetConfig,
    generate_scm_dataset,
    load_scm_dataset,
)
from inga.scm.dataset_core import CausalQueryConfig as CoreCausalQueryConfig
from inga.scm.random import RandomSCMConfig
from inga.scm.variable.base import Variable
from inga.scm.variable.gaussian import GaussianVariable


class ShiftVariable(GaussianVariable):
    """Simple custom variable used to validate SCM (de)serialization extension."""

    def __init__(
        self,
        name: str,
        sigma: float,
        shift: float,
        parent_names: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, sigma=sigma, parent_names=parent_names)
        self.shift = shift

    def f_mean(self, parents: dict[str, torch.Tensor]) -> torch.Tensor:
        if not parents:
            return torch.tensor(self.shift)
        reference = next(iter(parents.values()))
        return torch.zeros_like(reference) + self.shift


def _serialize_shift_variable(var: Variable) -> dict[str, object]:
    assert isinstance(var, ShiftVariable)
    return {
        "name": var.name,
        "sigma": var.sigma,
        "parent_names": list(var.parent_names),
        "shift": var.shift,
    }


class TestSCMDataset:
    """Unit tests for SCM dataset generation."""

    def test_dataset_shapes_and_query_rules(self) -> None:
        """Ensure dataset matches schema and query constraints."""
        config = SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.5, seed=1),
            num_samples=200,
            num_queries=5,
            min_observed=1,
            seed=123,
        )
        dataset = generate_scm_dataset(config)

        assert len(dataset.data) == 5
        for values in dataset.data.values():
            assert values.shape == (config.num_samples,)

        assert len(dataset.queries) >= config.num_queries
        for query in dataset.queries:
            assert query.treatment_name in query.observed_names
            assert query.outcome_name not in query.observed_names

        assert len(dataset.causal_effects) == len(dataset.queries)
        assert len(dataset.causal_biases) == len(dataset.queries)

        for key, effect in dataset.causal_effects.items():
            assert effect.shape == (config.num_samples,)
            assert key in dataset.causal_biases

    def test_dataset_default_expands_to_all_observed_treatments(self) -> None:
        """Default behavior should annotate all observed vars as candidate treatments."""
        config = SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.5, seed=3),
            num_samples=64,
            num_queries=2,
            min_observed=2,
            seed=5,
        )
        dataset = generate_scm_dataset(config)

        # Group by (outcome, observed-set) and verify one query per observed treatment.
        groups: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        for query in dataset.queries:
            key = (query.outcome_name, tuple(query.observed_names))
            groups.setdefault(key, set()).add(query.treatment_name)

        assert len(groups) == config.num_queries
        for (_, observed_names), treatments in groups.items():
            assert treatments == set(observed_names)

    def test_dataset_respects_explicit_treatment_name(self) -> None:
        """When treatment is provided, no all-observed expansion should occur."""
        config = SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.5, seed=7),
            num_samples=64,
            num_queries=3,
            min_observed=2,
            seed=11,
            treatment_name="X1",
        )
        dataset = generate_scm_dataset(config)

        assert len(dataset.queries) == config.num_queries
        assert len(dataset.causal_effects) == config.num_queries
        assert len(dataset.causal_biases) == config.num_queries
        for query in dataset.queries:
            assert query.treatment_name == "X1"
            assert "X1" in query.observed_names
            assert query.outcome_name != "X1"

    def test_invalid_dataset_config(self) -> None:
        """Ensure invalid configs raise errors."""
        with pytest.raises(ValueError, match="num_samples"):
            generate_scm_dataset(
                SCMDatasetConfig(
                    scm_config=RandomSCMConfig(num_variables=3),
                    num_samples=0,
                )
            )

        with pytest.raises(ValueError, match="num_queries"):
            generate_scm_dataset(
                SCMDatasetConfig(
                    scm_config=RandomSCMConfig(num_variables=3),
                    num_queries=0,
                )
            )

    def test_dataset_save_load_round_trip(self, tmp_path: Path) -> None:
        """Ensure datasets can be saved and loaded with SCM spec."""
        config = SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=4, parent_prob=0.5, seed=2),
            num_samples=100,
            num_queries=3,
            min_observed=1,
            seed=10,
        )
        dataset = generate_scm_dataset(config)
        target = tmp_path / "scm_dataset"
        dataset.save(target)

        loaded = load_scm_dataset(target)
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

    def test_dataset_load_supports_custom_variable_deserializers(
        self, tmp_path: Path
    ) -> None:
        """Custom variable classes can be serialized/deserialized via extension hooks."""
        custom = ShiftVariable(name="U", sigma=1.0, shift=0.3)
        child = LinearVariable(
            name="X",
            parent_names=["U"],
            sigma=1.0,
            coefs={"U": 1.0},
            intercept=0.0,
        )
        scm = SCM(variables=[custom, child])
        data = scm.generate(12)

        dataset = SCMDataset(
            scm=scm,
            data=data,
            queries=[],
            causal_effects={},
            causal_biases={},
        )
        target = tmp_path / "scm_dataset_custom"

        dataset.save(
            target,
            variable_serializers={
                ShiftVariable: _serialize_shift_variable,
            },
        )

        class_path = f"{ShiftVariable.__module__}.{ShiftVariable.__qualname__}"
        loaded = load_scm_dataset(
            target,
            variable_deserializers={
                class_path: lambda payload: ShiftVariable(
                    name=payload["name"],
                    sigma=payload["sigma"],
                    shift=payload["shift"],
                    parent_names=payload.get("parent_names"),
                )
            },
        )

        loaded_custom = loaded.scm._variables["U"]
        assert isinstance(loaded_custom, ShiftVariable)
        assert loaded_custom.shift == pytest.approx(0.3)

    def test_generate_dataset_from_user_provided_scm(self) -> None:
        """Users can generate datasets from a specific (non-random) SCM."""
        scm = SCM(
            variables=[
                LinearVariable(
                    name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
                LinearVariable(
                    name="X",
                    parent_names=["Z"],
                    sigma=1.0,
                    coefs={"Z": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    name="Y",
                    parent_names=["X"],
                    sigma=1.0,
                    coefs={"X": 2.0},
                    intercept=0.0,
                ),
            ]
        )

        dataset = scm.generate_dataset(
            num_samples=64,
            num_queries=2,
            min_observed=1,
            seed=7,
            queries=[
                CoreCausalQueryConfig(
                    treatment_name="X",
                    outcome_name="Y",
                    observed_names=["X"],
                ),
                CoreCausalQueryConfig(
                    treatment_name="X",
                    outcome_name="Z",
                    observed_names=["X", "Y"],
                ),
            ],
        )

        assert dataset.scm is scm
        assert set(dataset.data.keys()) == {"Z", "X", "Y"}
        for values in dataset.data.values():
            assert values.shape == (64,)

        assert len(dataset.queries) == 2
        assert len(dataset.causal_effects) == 2
        assert len(dataset.causal_biases) == 2
        assert dataset.queries[0] == CoreCausalQueryConfig(
            treatment_name="X",
            outcome_name="Y",
            observed_names=["X"],
        )
        assert dataset.queries[1] == CoreCausalQueryConfig(
            treatment_name="X",
            outcome_name="Z",
            observed_names=["X", "Y"],
        )
