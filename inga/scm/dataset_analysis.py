"""Utilities to inspect SCM datasets and report coverage-oriented diagnostics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from inga.scm.dataset import SCMDataset
from inga.scm.variable.gaussian import GaussianVariable


@dataclass(frozen=True)
class DatasetCoverageReport:
    """Coverage-oriented summary for a ``SCMDataset``."""

    num_samples: int
    num_variables: int
    num_queries: int
    gaussian_variables: list[str]
    observed_variable_coverage: float
    treatment_variable_coverage: float
    outcome_variable_coverage: float
    treatment_outcome_pair_coverage: float
    treatment_counts: dict[str, int]
    outcome_counts: dict[str, int]
    observed_counts: dict[str, int]
    missing_effect_keys: list[tuple[str, str, tuple[str, ...]]]
    missing_bias_keys: list[tuple[str, str, tuple[str, ...]]]
    extra_effect_keys: list[tuple[str, str, tuple[str, ...]]]
    extra_bias_keys: list[tuple[str, str, tuple[str, ...]]]
    nan_fraction_data: dict[str, float]
    nan_fraction_effects: dict[str, float]
    nan_fraction_biases: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert report into a plain JSON-serializable dictionary."""
        return {
            "num_samples": self.num_samples,
            "num_variables": self.num_variables,
            "num_queries": self.num_queries,
            "gaussian_variables": list(self.gaussian_variables),
            "coverage": {
                "observed_variable_coverage": self.observed_variable_coverage,
                "treatment_variable_coverage": self.treatment_variable_coverage,
                "outcome_variable_coverage": self.outcome_variable_coverage,
                "treatment_outcome_pair_coverage": self.treatment_outcome_pair_coverage,
            },
            "counts": {
                "treatment_counts": dict(self.treatment_counts),
                "outcome_counts": dict(self.outcome_counts),
                "observed_counts": dict(self.observed_counts),
            },
            "query_key_consistency": {
                "missing_effect_keys": [list(_serialize_key(key)) for key in self.missing_effect_keys],
                "missing_bias_keys": [list(_serialize_key(key)) for key in self.missing_bias_keys],
                "extra_effect_keys": [list(_serialize_key(key)) for key in self.extra_effect_keys],
                "extra_bias_keys": [list(_serialize_key(key)) for key in self.extra_bias_keys],
            },
            "nan_fraction": {
                "data": dict(self.nan_fraction_data),
                "effects": dict(self.nan_fraction_effects),
                "biases": dict(self.nan_fraction_biases),
            },
        }


def analyze_scm_dataset(dataset: SCMDataset) -> DatasetCoverageReport:
    """Analyze an SCM dataset and compute coverage/consistency diagnostics."""
    if not dataset.data:
        raise ValueError("Dataset has no variables in `data`.")

    sample_sizes = {len(values) for values in dataset.data.values()}
    if len(sample_sizes) != 1:
        raise ValueError("All dataset variables must have the same number of samples.")
    num_samples = sample_sizes.pop()

    variable_names = list(dataset.data.keys())
    num_variables = len(variable_names)

    gaussian_variables = [
        name
        for name, variable in dataset.scm._variables.items()
        if isinstance(variable, GaussianVariable)
    ]
    gaussian_variable_set = set(gaussian_variables)

    treatment_counter = Counter(query.treatment_name for query in dataset.queries)
    outcome_counter = Counter(query.outcome_name for query in dataset.queries)
    observed_counter = Counter(
        observed_name
        for query in dataset.queries
        for observed_name in query.observed_names
    )

    observed_covered = set(observed_counter)
    treatment_covered = set(treatment_counter)
    outcome_covered = set(outcome_counter)

    variable_name_set = set(variable_names)
    observed_variable_coverage = _safe_ratio(
        len(observed_covered & variable_name_set),
        num_variables,
    )
    treatment_variable_coverage = _safe_ratio(
        len(treatment_covered & variable_name_set),
        num_variables,
    )
    outcome_variable_coverage = _safe_ratio(
        len(outcome_covered & variable_name_set),
        num_variables,
    )

    observed_pairs = {
        (query.treatment_name, query.outcome_name)
        for query in dataset.queries
        if query.treatment_name in gaussian_variable_set
        and query.outcome_name in gaussian_variable_set
        and query.treatment_name != query.outcome_name
    }
    possible_pairs = max(0, len(gaussian_variables) * (len(gaussian_variables) - 1))
    treatment_outcome_pair_coverage = _safe_ratio(len(observed_pairs), possible_pairs)

    expected_keys = {
        (query.treatment_name, query.outcome_name, tuple(query.observed_names))
        for query in dataset.queries
    }
    effect_keys = set(dataset.causal_effects.keys())
    bias_keys = set(dataset.causal_biases.keys())

    missing_effect_keys = sorted(expected_keys - effect_keys)
    missing_bias_keys = sorted(expected_keys - bias_keys)
    extra_effect_keys = sorted(effect_keys - expected_keys)
    extra_bias_keys = sorted(bias_keys - expected_keys)

    return DatasetCoverageReport(
        num_samples=num_samples,
        num_variables=num_variables,
        num_queries=len(dataset.queries),
        gaussian_variables=gaussian_variables,
        observed_variable_coverage=observed_variable_coverage,
        treatment_variable_coverage=treatment_variable_coverage,
        outcome_variable_coverage=outcome_variable_coverage,
        treatment_outcome_pair_coverage=treatment_outcome_pair_coverage,
        treatment_counts=dict(treatment_counter),
        outcome_counts=dict(outcome_counter),
        observed_counts=dict(observed_counter),
        missing_effect_keys=missing_effect_keys,
        missing_bias_keys=missing_bias_keys,
        extra_effect_keys=extra_effect_keys,
        extra_bias_keys=extra_bias_keys,
        nan_fraction_data={
            name: _nan_fraction(values) for name, values in dataset.data.items()
        },
        nan_fraction_effects={
            str(_serialize_key(key)): _nan_fraction(values)
            for key, values in dataset.causal_effects.items()
        },
        nan_fraction_biases={
            str(_serialize_key(key)): _nan_fraction(values)
            for key, values in dataset.causal_biases.items()
        },
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _nan_fraction(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.isnan(values).sum().item()) / float(values.numel())


def _serialize_key(key: tuple[str, str, tuple[str, ...]]) -> tuple[str, str, list[str]]:
    return key[0], key[1], list(key[2])
