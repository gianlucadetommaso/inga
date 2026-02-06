"""Dataset generation utilities for SEMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from steindag.sem.base import SEM
from steindag.sem.random import RandomSEMConfig, random_sem, resolve_transforms
from steindag.variable.base import Variable
from steindag.variable.linear import LinearVariable
from steindag.variable.functional import FunctionalVariable


@dataclass(frozen=True)
class CausalQueryConfig:
    """Configuration for a causal query within a dataset."""

    treatment_name: str
    outcome_name: str
    observed_names: list[str]


@dataclass(frozen=True)
class SEMDataset:
    """Dataset containing samples and causal query results."""

    sem: SEM
    data: dict[str, Tensor]
    queries: list[CausalQueryConfig]
    causal_effects: dict[tuple[str, str, tuple[str, ...]], Tensor]
    causal_biases: dict[tuple[str, str, tuple[str, ...]], Tensor]

    def save(self, path: str | Path) -> None:
        """Save dataset to disk (JSON metadata + tensor blob)."""
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "sem": _serialize_sem(self.sem),
            "queries": [
                {
                    "treatment_name": query.treatment_name,
                    "outcome_name": query.outcome_name,
                    "observed_names": query.observed_names,
                }
                for query in self.queries
            ],
            "effect_keys": [
                [key[0], key[1], list(key[2])] for key in self.causal_effects
            ],
            "bias_keys": [[key[0], key[1], list(key[2])] for key in self.causal_biases],
        }
        base_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

        arrays: dict[str, Tensor] = {}
        for name, values in self.data.items():
            arrays[f"data::{name}"] = values

        for idx, key in enumerate(self.causal_effects):
            arrays[f"effect::{idx}"] = self.causal_effects[key]
        for idx, key in enumerate(self.causal_biases):
            arrays[f"bias::{idx}"] = self.causal_biases[key]

        torch.save(arrays, base_path.with_suffix(".pt"))


@dataclass(frozen=True)
class SEMDatasetConfig:
    """Configuration for SEM dataset generation."""

    sem_config: RandomSEMConfig
    num_samples: int = 1000
    num_queries: int = 10
    min_observed: int = 1
    seed: int | None = None


def _sample_query(
    rng: random.Random,
    variable_names: list[str],
    min_observed: int,
) -> CausalQueryConfig:
    """Sample a treatment/outcome/observed configuration.

    Treatment is always included in observed; outcome is never observed.
    """
    if len(variable_names) < 2:
        raise ValueError("At least two variables are required for a causal query.")

    treatment_name = rng.choice(variable_names)
    candidate_outcomes = [name for name in variable_names if name != treatment_name]
    outcome_name = rng.choice(candidate_outcomes)

    observed_candidates = [
        name for name in variable_names if name not in {treatment_name, outcome_name}
    ]
    max_observed = max(min_observed, len(observed_candidates))
    observed_size = rng.randint(min_observed, max_observed)
    observed_subset = rng.sample(observed_candidates, k=observed_size)
    observed_names = [treatment_name, *observed_subset]

    return CausalQueryConfig(
        treatment_name=treatment_name,
        outcome_name=outcome_name,
        observed_names=observed_names,
    )


def generate_sem_dataset(config: SEMDatasetConfig) -> SEMDataset:
    """Generate a SEM dataset with multiple causal queries."""
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if config.num_queries <= 0:
        raise ValueError("num_queries must be positive.")

    rng = random.Random(config.seed)
    sem = random_sem(config.sem_config)
    data = sem.generate(config.num_samples)
    variable_names = list(sem._variables.keys())

    queries = [
        _sample_query(
            rng,
            variable_names=variable_names,
            min_observed=config.min_observed,
        )
        for _ in range(config.num_queries)
    ]

    causal_effects: dict[tuple[str, str, tuple[str, ...]], Tensor] = {}
    causal_biases: dict[tuple[str, str, tuple[str, ...]], Tensor] = {}

    for query in queries:
        observed = {name: data[name] for name in query.observed_names}
        sem.posterior.fit(observed)
        effect = sem.causal_effect(
            observed,
            treatment_name=query.treatment_name,
            outcome_name=query.outcome_name,
        )
        bias = sem.causal_bias(
            observed,
            treatment_name=query.treatment_name,
            outcome_name=query.outcome_name,
        )
        key = (
            query.treatment_name,
            query.outcome_name,
            tuple(query.observed_names),
        )
        causal_effects[key] = effect
        causal_biases[key] = bias

    return SEMDataset(
        sem=sem,
        data=data,
        queries=queries,
        causal_effects=causal_effects,
        causal_biases=causal_biases,
    )


def load_sem_dataset(path: str | Path) -> SEMDataset:
    """Load a SEM dataset from disk."""
    base_path = Path(path)
    metadata = json.loads(base_path.with_suffix(".json").read_text())
    arrays = torch.load(base_path.with_suffix(".pt"))

    sem = _deserialize_sem(metadata["sem"])
    data = {
        key.split("::", 1)[1]: value
        for key, value in arrays.items()
        if key.startswith("data::")
    }

    queries = [
        CausalQueryConfig(
            treatment_name=item["treatment_name"],
            outcome_name=item["outcome_name"],
            observed_names=list(item["observed_names"]),
        )
        for item in metadata["queries"]
    ]

    causal_effects = {
        (key[0], key[1], tuple(key[2])): arrays[f"effect::{idx}"]
        for idx, key in enumerate(metadata["effect_keys"])
    }
    causal_biases = {
        (key[0], key[1], tuple(key[2])): arrays[f"bias::{idx}"]
        for idx, key in enumerate(metadata["bias_keys"])
    }

    return SEMDataset(
        sem=sem,
        data=data,
        queries=queries,
        causal_effects=causal_effects,
        causal_biases=causal_biases,
    )


def _serialize_sem(sem: SEM) -> dict[str, Any]:
    """Serialize SEM definition to a JSON-compatible spec."""
    variables = []
    for name, variable in sem._variables.items():
        if isinstance(variable, LinearVariable):
            variables.append(
                {
                    "type": "linear",
                    "name": name,
                    "parent_names": list(variable.parent_names),
                    "sigma": variable.sigma,
                    "coefs": dict(variable._coefs),
                    "intercept": float(variable._intercept),
                }
            )
        elif isinstance(variable, FunctionalVariable):
            variables.append(
                {
                    "type": "functional",
                    "name": name,
                    "parent_names": list(variable.parent_names),
                    "sigma": variable.sigma,
                    "coefs": dict(variable._coefs or {}),
                    "intercept": float(variable._intercept or 0.0),
                    "transforms": list(variable._transforms or []),
                }
            )
        else:
            raise ValueError(f"Unsupported variable type: {type(variable)}")

    return {"variables": variables}


def _deserialize_sem(spec: dict[str, Any]) -> SEM:
    """Reconstruct a SEM from a serialized spec."""
    variables_spec = spec["variables"]
    if not isinstance(variables_spec, list):
        raise ValueError("SEM spec 'variables' must be a list.")

    variables: list[Variable] = []
    for item in variables_spec:
        if not isinstance(item, dict):
            raise ValueError("SEM variable spec entries must be dictionaries.")
        var_type = item.get("type")
        if var_type == "linear":
            variables.append(
                LinearVariable(
                    name=item["name"],
                    parent_names=item["parent_names"],
                    sigma=item["sigma"],
                    coefs=item["coefs"],
                    intercept=item["intercept"],
                )
            )
        elif var_type == "functional":
            transforms = resolve_transforms(item["transforms"])
            f_mean = _build_f_mean_from_spec(
                item["parent_names"],
                item["coefs"],
                item["intercept"],
                transforms,
            )
            variables.append(
                FunctionalVariable(
                    name=item["name"],
                    parent_names=item["parent_names"],
                    sigma=item["sigma"],
                    f_mean=f_mean,
                    coefs=item["coefs"],
                    intercept=item["intercept"],
                    transforms=item["transforms"],
                )
            )
        else:
            raise ValueError(f"Unsupported variable type '{var_type}'.")

    return SEM(variables=variables)


def _build_f_mean_from_spec(
    parent_names: list[str],
    coefs: dict[str, float],
    intercept: float,
    transforms: Sequence[Callable[[Tensor], Tensor]],
) -> Callable[[dict[str, Tensor]], Tensor]:
    def f_mean(parents: dict[str, Tensor]) -> Tensor:
        base = torch.tensor(intercept)
        for parent_name in parent_names:
            base = base + coefs[parent_name] * parents[parent_name]
        if transforms:
            for transform in transforms:
                base = transform(base)
        return base

    return f_mean
