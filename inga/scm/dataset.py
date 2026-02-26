"""Dataset generation utilities for SEMs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
from torch import Tensor

from inga.scm.random import RandomSCMConfig, random_scm, resolve_transforms
from inga.scm.variable.base import Variable
from inga.scm.variable.gaussian import GaussianVariable
from inga.scm.variable.linear import LinearVariable
from inga.scm.variable.functional import FunctionalVariable

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

    scm_config: RandomSCMConfig
    num_samples: int = 1000
    num_queries: int = 10
    min_observed: int = 1
    seed: int | None = None
    treatment_name: str | None = None


def _sample_query(
    rng: random.Random,
    variable_names: list[str],
    queryable_names: list[str],
    min_observed: int,
    treatment_name: str | None = None,
) -> CausalQueryConfig:
    """Sample a treatment/outcome/observed configuration.

    Treatment is always included in observed; outcome is never observed.
    """
    if len(variable_names) < 2:
        raise ValueError("At least two variables are required for a causal query.")

    if len(queryable_names) < 2:
        raise ValueError(
            "At least two Gaussian/queryable variables are required for a causal query."
        )

    if treatment_name is not None and treatment_name not in queryable_names:
        raise ValueError(f"Unknown treatment_name '{treatment_name}'.")

    sampled_treatment_name = (
        treatment_name if treatment_name is not None else rng.choice(queryable_names)
    )
    candidate_outcomes = [
        name for name in queryable_names if name != sampled_treatment_name
    ]
    if not candidate_outcomes:
        raise ValueError(
            "Could not sample an outcome variable distinct from treatment."
        )
    outcome_name = rng.choice(candidate_outcomes)

    observed_candidates = [
        name
        for name in variable_names
        if name != outcome_name and name != sampled_treatment_name
    ]
    required_additional_observed = max(0, min_observed - 1)
    if required_additional_observed > len(observed_candidates):
        raise ValueError(
            "min_observed is too large for available variables in this query setup."
        )
    additional_observed_size = rng.randint(
        required_additional_observed,
        len(observed_candidates),
    )
    observed_subset = rng.sample(observed_candidates, k=additional_observed_size)
    observed_names = [sampled_treatment_name, *observed_subset]

    return CausalQueryConfig(
        treatment_name=sampled_treatment_name,
        outcome_name=outcome_name,
        observed_names=observed_names,
    )


def _query_key(query: CausalQueryConfig) -> tuple[str, str, tuple[str, ...]]:
    return (query.treatment_name, query.outcome_name, tuple(query.observed_names))


def generate_scm_dataset(config: SCMDatasetConfig) -> SCMDataset:
    """Generate a SCM dataset with multiple causal queries."""
    if config.num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if config.num_queries <= 0:
        raise ValueError("num_queries must be positive.")

    scm = random_scm(config.scm_config)
    return _generate_dataset_from_scm(
        scm=scm,
        num_samples=config.num_samples,
        num_queries=config.num_queries,
        min_observed=config.min_observed,
        seed=config.seed,
        treatment_name=config.treatment_name,
    )


def generate_dataset_from_scm(
    scm: SCM,
    num_samples: int = 1000,
    num_queries: int | None = None,
    min_observed: int = 1,
    seed: int | None = None,
    treatment_name: str | None = None,
    queries: list[CausalQueryConfig] | None = None,
) -> SCMDataset:
    """Generate a dataset from a user-provided (fixed) SCM.

    This is useful when you want full control over the causal graph and
    structural equations instead of sampling a random SCM.

    If ``queries`` is provided, those queries are used as-is (no sampling).
    In that case ``num_queries`` is optional and, when provided, must match
    ``len(queries)``.
    """
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
    queryable_names = [
        name
        for name, variable in scm._variables.items()
        if isinstance(variable, GaussianVariable)
    ]

    if len(queryable_names) < 2:
        raise ValueError(
            "Dataset generation requires at least two Gaussian variables to define "
            "scalar treatment/outcome causal queries."
        )
    if treatment_name is not None and treatment_name not in queryable_names:
        raise ValueError(
            "treatment_name must refer to a Gaussian variable when generating "
            "causal-query datasets."
        )

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
                queryable_names=queryable_names,
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
            if query.treatment_name not in queryable_names:
                raise ValueError(
                    "Provided treatment_name must refer to a Gaussian/queryable variable."
                )
            if query.outcome_name not in variable_names:
                raise ValueError(
                    f"Unknown outcome_name '{query.outcome_name}' in provided queries."
                )
            if query.outcome_name not in queryable_names:
                raise ValueError(
                    "Provided outcome_name must refer to a Gaussian/queryable variable."
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

        # By default, synthesize treatment-specific causal effects/biases for
        # all observed variables so each input feature has causal annotations.
        for observed_treatment_name in query.observed_names:
            if observed_treatment_name not in queryable_names:
                continue
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


def load_scm_dataset(
    path: str | Path,
    variable_deserializers: dict[str, Callable[[dict[str, Any]], Variable]]
    | None = None,
) -> SCMDataset:
    """Load a SCM dataset from disk."""
    base_path = Path(path)
    metadata = json.loads(base_path.with_suffix(".json").read_text())
    arrays = torch.load(base_path.with_suffix(".pt"))

    scm = _deserialize_scm(
        metadata["scm"],
        variable_deserializers=variable_deserializers,
    )
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


def _deserialize_scm(
    spec: dict[str, Any],
    variable_deserializers: dict[str, Callable[[dict[str, Any]], Variable]]
    | None = None,
) -> SCM:
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
        elif var_type == "custom":
            class_path = item.get("class_path")
            payload = item.get("payload")
            if not isinstance(class_path, str) or not class_path:
                raise ValueError("Custom variable specs must include 'class_path'.")
            if not isinstance(payload, dict):
                raise ValueError("Custom variable specs must include dict 'payload'.")

            variable = _deserialize_custom_variable(
                class_path=class_path,
                payload=payload,
                variable_deserializers=variable_deserializers,
            )
            variables.append(variable)
        else:
            raise ValueError(f"Unsupported variable type '{var_type}'.")

    scm_cls = getattr(importlib.import_module("inga.scm.base"), "SCM")
    return scm_cls(variables=variables)


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


def _class_path(variable_cls: type[Any]) -> str:
    """Return importable fully qualified class path."""
    return f"{variable_cls.__module__}.{variable_cls.__qualname__}"


def _deserialize_custom_variable(
    class_path: str,
    payload: dict[str, Any],
    variable_deserializers: dict[str, Callable[[dict[str, Any]], Variable]]
    | None = None,
) -> Variable:
    if variable_deserializers is not None and class_path in variable_deserializers:
        variable = variable_deserializers[class_path](payload)
    else:
        variable_cls = _import_variable_class(class_path)
        from_dict = getattr(variable_cls, "from_dict", None)
        if callable(from_dict):
            variable = from_dict(payload)
        else:
            variable = variable_cls(**payload)

    if not isinstance(variable, Variable):
        raise ValueError(
            f"Custom variable deserializer for '{class_path}' must return Variable, "
            f"got {type(variable)}."
        )
    return variable


def _import_variable_class(class_path: str) -> type[Variable]:
    """Import a variable class from '<module>.<ClassName>' path."""
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"Invalid custom variable class_path '{class_path}'. "
            "Expected '<module>.<ClassName>'."
        )

    module = importlib.import_module(module_name)
    variable_cls = getattr(module, class_name, None)
    if not isinstance(variable_cls, type) or not issubclass(variable_cls, Variable):
        raise ValueError(
            f"Custom variable class '{class_path}' is not a Variable subclass."
        )
    return variable_cls
