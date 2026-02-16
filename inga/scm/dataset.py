"""Dataset generation utilities for SEMs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from inga.scm.base import SCM
from inga.scm.random import RandomSCMConfig, random_scm, resolve_transforms
from inga.variable.base import Variable
from inga.variable.linear import LinearVariable
from inga.variable.functional import FunctionalVariable


@dataclass(frozen=True)
class CausalQueryConfig:
    """Configuration for a causal query within a dataset."""

    treatment_name: str
    outcome_name: str
    observed_names: list[str]


@dataclass(frozen=True)
class SCMDataset:
    """Dataset containing samples and causal query results."""

    scm: SCM
    data: dict[str, Tensor]
    queries: list[CausalQueryConfig]
    causal_effects: dict[tuple[str, str, tuple[str, ...]], Tensor]
    causal_biases: dict[tuple[str, str, tuple[str, ...]], Tensor]

    def save(self, path: str | Path) -> None:
        """Save dataset to disk (JSON metadata + tensor blob)."""
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "scm": _serialize_scm(self.scm),
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
class SCMDatasetConfig:
    """Configuration for SCM dataset generation."""

    scm_config: RandomSCMConfig
    num_samples: int = 1000
    num_queries: int = 10
    min_observed: int = 1
    seed: int | None = None
    treatment_name: str | None = None


def _sample_query(
    rng: random.Random,
    variable_names: list[str],
    min_observed: int,
    treatment_name: str | None = None,
) -> CausalQueryConfig:
    """Sample a treatment/outcome/observed configuration.

    Treatment is always included in observed; outcome is never observed.
    """
    if len(variable_names) < 2:
        raise ValueError("At least two variables are required for a causal query.")

    if treatment_name is not None and treatment_name not in variable_names:
        raise ValueError(f"Unknown treatment_name '{treatment_name}'.")

    candidate_outcomes = (
        [name for name in variable_names if name != treatment_name]
        if treatment_name is not None
        else variable_names
    )
    outcome_name = rng.choice(candidate_outcomes)

    observed_candidates = [
        name
        for name in variable_names
        if name != outcome_name and (treatment_name is None or name != treatment_name)
    ]
    required_observed = (
        min_observed if treatment_name is None else max(0, min_observed - 1)
    )
    max_observed = max(required_observed, len(observed_candidates))
    lower_observed = min_observed if treatment_name is None else max(1, min_observed)
    observed_size = rng.randint(lower_observed, max_observed)
    if treatment_name is None:
        observed_names = rng.sample(observed_candidates, k=observed_size)
        sampled_treatment_name = observed_names[0]
    else:
        observed_subset = rng.sample(observed_candidates, k=max(0, observed_size - 1))
        observed_names = [treatment_name, *observed_subset]
        sampled_treatment_name = treatment_name

    return CausalQueryConfig(
        treatment_name=sampled_treatment_name,
        outcome_name=outcome_name,
        observed_names=observed_names,
    )


def generate_scm_dataset(config: SCMDatasetConfig) -> SCMDataset:
    """Generate a SCM dataset with multiple causal queries."""
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if config.num_queries <= 0:
        raise ValueError("num_queries must be positive.")

    rng = random.Random(config.seed)
    scm = random_scm(config.scm_config)
    data = scm.generate(config.num_samples)
    variable_names = list(scm._variables.keys())

    queries = [
        _sample_query(
            rng,
            variable_names=variable_names,
            min_observed=config.min_observed,
            treatment_name=config.treatment_name,
        )
        for _ in range(config.num_queries)
    ]

    expanded_queries: list[CausalQueryConfig] = []
    for query in queries:
        if config.treatment_name is not None:
            expanded_queries.append(query)
            continue

        # By default, synthesize treatment-specific causal effects/biases for
        # all observed variables so each input feature has causal annotations.
        for treatment_name in query.observed_names:
            expanded_queries.append(
                CausalQueryConfig(
                    treatment_name=treatment_name,
                    outcome_name=query.outcome_name,
                    observed_names=list(query.observed_names),
                )
            )

    causal_effects: dict[tuple[str, str, tuple[str, ...]], Tensor] = {}
    causal_biases: dict[tuple[str, str, tuple[str, ...]], Tensor] = {}

    for query in expanded_queries:
        observed = {name: data[name] for name in query.observed_names}
        scm.posterior.fit(observed)
        effect = scm.causal_effect(
            observed,
            treatment_name=query.treatment_name,
            outcome_name=query.outcome_name,
        )
        bias = scm.causal_bias(
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

    return SCMDataset(
        scm=scm,
        data=data,
        queries=expanded_queries,
        causal_effects=causal_effects,
        causal_biases=causal_biases,
    )


def load_scm_dataset(path: str | Path) -> SCMDataset:
    """Load a SCM dataset from disk."""
    base_path = Path(path)
    metadata = json.loads(base_path.with_suffix(".json").read_text())
    arrays = torch.load(base_path.with_suffix(".pt"))

    scm = _deserialize_scm(metadata["scm"])
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

    return SCMDataset(
        scm=scm,
        data=data,
        queries=queries,
        causal_effects=causal_effects,
        causal_biases=causal_biases,
    )


def _serialize_scm(scm: SCM) -> dict[str, Any]:
    """Serialize SCM definition to a JSON-compatible spec."""
    variables = []
    for name, variable in scm._variables.items():
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


def _deserialize_scm(spec: dict[str, Any]) -> SCM:
    """Reconstruct a SCM from a serialized spec."""
    variables_spec = spec["variables"]
    if not isinstance(variables_spec, list):
        raise ValueError("SCM spec 'variables' must be a list.")

    variables: list[Variable] = []
    for item in variables_spec:
        if not isinstance(item, dict):
            raise ValueError("SCM variable spec entries must be dictionaries.")
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

    return SCM(variables=variables)


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
