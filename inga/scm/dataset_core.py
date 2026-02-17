"""Core dataset generation utilities for SCMs.

This module is intentionally independent from random SCM construction and
serialization so it can be imported from ``inga.scm.base`` without creating
import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Callable

import torch
from torch import Tensor

from inga.scm.variable.base import Variable
from inga.scm.variable.functional import FunctionalVariable
from inga.scm.variable.linear import LinearVariable

if TYPE_CHECKING:
    from inga.scm.base import SCM


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

    def save(
        self,
        path: str | Path,
        variable_serializers: dict[type[Variable], Callable[[Variable], dict[str, Any]]]
        | None = None,
    ) -> None:
        """Save dataset to disk (JSON metadata + tensor blob)."""
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "scm": _serialize_scm(
                self.scm,
                variable_serializers=variable_serializers,
            ),
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

    scm_config: object
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


def _query_key(query: CausalQueryConfig) -> tuple[str, str, tuple[str, ...]]:
    return (query.treatment_name, query.outcome_name, tuple(query.observed_names))


def generate_dataset_from_scm(
    scm: SCM,
    num_samples: int = 1000,
    num_queries: int | None = None,
    min_observed: int = 1,
    seed: int | None = None,
    treatment_name: str | None = None,
    queries: list[CausalQueryConfig] | None = None,
) -> SCMDataset:
    """Generate a dataset from a user-provided (fixed) SCM."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if num_queries is not None and num_queries <= 0:
        raise ValueError("num_queries must be positive.")

    if queries is not None:
        if num_queries is not None and num_queries != len(queries):
            raise ValueError(
                "When 'queries' is provided, 'num_queries' must match len(queries)."
            )
        effective_num_queries = len(queries)
    else:
        effective_num_queries = 10 if num_queries is None else num_queries

    return _generate_dataset_from_scm(
        scm=scm,
        num_samples=num_samples,
        num_queries=effective_num_queries,
        min_observed=min_observed,
        seed=seed,
        treatment_name=treatment_name,
        queries=queries,
    )


def _generate_dataset_from_scm(
    scm: SCM,
    num_samples: int,
    num_queries: int,
    min_observed: int,
    seed: int | None,
    treatment_name: str | None,
    queries: list[CausalQueryConfig] | None = None,
) -> SCMDataset:
    rng = random.Random(seed)
    data = scm.generate(num_samples)
    variable_names = list(scm._variables.keys())

    sampled_queries: list[CausalQueryConfig]
    if queries is None:
        sampled_queries = []
        seen_query_keys: set[tuple[str, str, tuple[str, ...]]] = set()
        attempts = 0
        max_attempts = max(100, num_queries * 50)
        while len(sampled_queries) < num_queries and attempts < max_attempts:
            attempts += 1
            query = _sample_query(
                rng,
                variable_names=variable_names,
                min_observed=min_observed,
                treatment_name=treatment_name,
            )
            key = _query_key(query)
            if key in seen_query_keys:
                continue
            seen_query_keys.add(key)
            sampled_queries.append(query)

        if len(sampled_queries) < num_queries:
            raise ValueError(
                "Could not sample enough unique causal queries for this SCM/config. "
                "Try reducing num_queries or min_observed."
            )
    else:
        sampled_queries = list(queries)
        for query in sampled_queries:
            if query.treatment_name not in variable_names:
                raise ValueError(
                    f"Unknown treatment_name '{query.treatment_name}' in provided queries."
                )
            if query.outcome_name not in variable_names:
                raise ValueError(
                    f"Unknown outcome_name '{query.outcome_name}' in provided queries."
                )
            if query.outcome_name in query.observed_names:
                raise ValueError("Outcome must not be in observed_names.")
            if query.treatment_name not in query.observed_names:
                raise ValueError("Treatment must be included in observed_names.")
            for observed_name in query.observed_names:
                if observed_name not in variable_names:
                    raise ValueError(
                        f"Unknown observed variable '{observed_name}' in provided queries."
                    )

    expanded_queries: list[CausalQueryConfig] = []
    for query in sampled_queries:
        if treatment_name is not None or queries is not None:
            expanded_queries.append(query)
            continue

        for observed_treatment_name in query.observed_names:
            expanded_queries.append(
                CausalQueryConfig(
                    treatment_name=observed_treatment_name,
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


def _serialize_scm(
    scm: SCM,
    variable_serializers: dict[type[Variable], Callable[[Variable], dict[str, Any]]]
    | None = None,
) -> dict[str, Any]:
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
            custom_payload: dict[str, Any] | None = None
            if variable_serializers is not None:
                for variable_cls, serializer in variable_serializers.items():
                    if isinstance(variable, variable_cls):
                        custom_payload = serializer(variable)
                        break

            if custom_payload is None:
                to_dict = getattr(variable, "to_dict", None)
                if callable(to_dict):
                    custom_payload = to_dict()

            if custom_payload is None:
                # Best-effort fallback for simple Variable subclasses that rely on
                # the base constructor signature (name, sigma, parent_names).
                custom_payload = {
                    "name": variable.name,
                    "sigma": variable.sigma,
                    "parent_names": list(variable.parent_names),
                }

            if not isinstance(custom_payload, dict):
                raise ValueError(
                    f"Unsupported variable type: {type(variable)}. "
                    "Provide a serializer via 'variable_serializers' when saving, "
                    "or implement a 'to_dict() -> dict' method on the variable class."
                )

            variables.append(
                {
                    "type": "custom",
                    "class_path": _class_path(type(variable)),
                    "payload": custom_payload,
                    "name": name,
                }
            )

    return {"variables": variables}


def _class_path(variable_cls: type[Any]) -> str:
    """Return importable fully qualified class path."""
    return f"{variable_cls.__module__}.{variable_cls.__qualname__}"
