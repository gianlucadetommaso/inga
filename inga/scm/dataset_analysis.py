"""Exhaustive analysis utilities for SCM datasets and collections."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from inga.scm.dataset import SCMDataset
from inga.scm.dataset_collection import SCMDatasetCollection
from inga.scm.variable.functional import FunctionalVariable
from inga.scm.variable.gaussian import GaussianVariable
from inga.scm.variable.linear import LinearVariable


@dataclass(frozen=True)
class DatasetCoverageReport:
    """Detailed report for one dataset."""

    num_samples: int
    num_variables: int
    num_queries: int
    num_edges: int
    variable_types: dict[str, str]
    variable_specs: dict[str, dict[str, Any]]
    parent_map: dict[str, list[str]]
    in_degree: dict[str, int]
    out_degree: dict[str, int]
    transform_usage: dict[str, int]
    observed_variable_coverage: float
    treatment_variable_coverage: float
    outcome_variable_coverage: float
    treatment_outcome_pair_coverage: float
    query_density_over_pairs: float
    observed_size_summary: dict[str, float]
    data_distribution: dict[str, dict[str, float]]
    pairwise_relations: dict[str, dict[str, float]]
    effect_distribution: dict[str, dict[str, float]]
    bias_distribution: dict[str, dict[str, float]]
    effect_bias_alignment: dict[str, dict[str, float]]
    treatment_counts: dict[str, int]
    outcome_counts: dict[str, int]
    observed_counts: dict[str, int]
    treatment_outcome_pair_counts: dict[str, int]
    missing_effect_keys: list[tuple[str, str, tuple[str, ...]]]
    missing_bias_keys: list[tuple[str, str, tuple[str, ...]]]
    extra_effect_keys: list[tuple[str, str, tuple[str, ...]]]
    extra_bias_keys: list[tuple[str, str, tuple[str, ...]]]
    nan_fraction_data: dict[str, float]
    nan_fraction_effects: dict[str, float]
    nan_fraction_biases: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {
            "coverage": {
                "observed_variable_coverage": self.observed_variable_coverage,
                "treatment_variable_coverage": self.treatment_variable_coverage,
                "outcome_variable_coverage": self.outcome_variable_coverage,
                "treatment_outcome_pair_coverage": self.treatment_outcome_pair_coverage,
                "query_density_over_pairs": self.query_density_over_pairs,
            },
            "counts": {
                "treatment_counts": dict(self.treatment_counts),
                "outcome_counts": dict(self.outcome_counts),
                "observed_counts": dict(self.observed_counts),
                "treatment_outcome_pair_counts": dict(
                    self.treatment_outcome_pair_counts
                ),
            },
            "distribution": {
                "data": dict(self.data_distribution),
                "effects": dict(self.effect_distribution),
                "biases": dict(self.bias_distribution),
            },
            "nan_fraction": {
                "data": dict(self.nan_fraction_data),
                "effects": dict(self.nan_fraction_effects),
                "biases": dict(self.nan_fraction_biases),
            },
            "missing_effect_keys": [
                list(_serialize_key(key)) for key in self.missing_effect_keys
            ],
            "missing_bias_keys": [
                list(_serialize_key(key)) for key in self.missing_bias_keys
            ],
            "extra_effect_keys": [
                list(_serialize_key(key)) for key in self.extra_effect_keys
            ],
            "extra_bias_keys": [
                list(_serialize_key(key)) for key in self.extra_bias_keys
            ],
        }


@dataclass(frozen=True)
class DatasetCollectionCoverageReport:
    """Detailed report for a dataset collection."""

    num_datasets: int
    per_dataset_reports: list[DatasetCoverageReport]
    aggregate_summary: dict[str, float]
    dataset_metric_summary: dict[str, dict[str, float]]
    structure_summary: dict[str, Any]
    domain_coverage_summary: dict[str, dict[str, float]]
    pair_relation_summary: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_datasets": self.num_datasets,
            "aggregate_summary": dict(self.aggregate_summary),
            "dataset_metric_summary": dict(self.dataset_metric_summary),
            "structure_summary": dict(self.structure_summary),
            "domain_coverage_summary": dict(self.domain_coverage_summary),
            "pair_relation_summary": dict(self.pair_relation_summary),
            "per_dataset_reports": [
                report.to_dict() for report in self.per_dataset_reports
            ],
        }


@dataclass(frozen=True)
class SimilarityThresholds:
    """Decision thresholds for real-vs-synthetic similarity checks."""

    max_marginal_distance: float = 0.50
    max_correlation_distance: float = 0.35
    max_mean_outside_fraction: float = 0.15
    min_pair_overlap: float = 0.10
    min_overall_score: float = 0.55


@dataclass(frozen=True)
class RealDatasetSimilarityReport:
    """Similarity report comparing a real dataset to a synthetic collection."""

    compared_variables: list[str]
    num_real_samples: int
    num_synthetic_samples_total: int
    marginal_distance_mean: float
    marginal_distance_max: float
    correlation_distance: float
    mean_outside_fraction: float
    max_outside_fraction: float
    treatment_outcome_pair_overlap: float
    overall_score: float
    similar_enough: bool
    variable_marginal_distances: dict[str, float]
    variable_outside_fractions: dict[str, float]
    thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compared_variables": list(self.compared_variables),
            "num_real_samples": self.num_real_samples,
            "num_synthetic_samples_total": self.num_synthetic_samples_total,
            "marginal_distance_mean": self.marginal_distance_mean,
            "marginal_distance_max": self.marginal_distance_max,
            "correlation_distance": self.correlation_distance,
            "mean_outside_fraction": self.mean_outside_fraction,
            "max_outside_fraction": self.max_outside_fraction,
            "treatment_outcome_pair_overlap": self.treatment_outcome_pair_overlap,
            "overall_score": self.overall_score,
            "similar_enough": self.similar_enough,
            "variable_marginal_distances": dict(self.variable_marginal_distances),
            "variable_outside_fractions": dict(self.variable_outside_fractions),
            "thresholds": dict(self.thresholds),
        }


def infer_similarity_key_factors(
    candidate_params: list[dict[str, float]],
    scores: list[float],
) -> list[tuple[str, float]]:
    """Rank candidate-generation factors by absolute linear association with score."""
    if not candidate_params or len(candidate_params) != len(scores):
        return []
    keys = sorted({key for params in candidate_params for key in params.keys()})
    if not keys:
        return []

    score_arr = np.array(scores, dtype=float)
    if score_arr.size <= 1:
        return [(key, 0.0) for key in keys]

    ranked: list[tuple[str, float]] = []
    for key in keys:
        values = np.array(
            [float(params.get(key, 0.0)) for params in candidate_params], dtype=float
        )
        if np.std(values) <= 1e-12 or np.std(score_arr) <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, score_arr)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        ranked.append((key, corr))

    ranked.sort(key=lambda item: abs(item[1]), reverse=True)
    return ranked


def analyze_scm_dataset(
    dataset: SCMDataset,
    *,
    max_pairwise_pairs: int = 40,
) -> DatasetCoverageReport:
    if not dataset.data:
        raise ValueError("Dataset has no variables in `data`.")

    sample_sizes = {len(values) for values in dataset.data.values()}
    if len(sample_sizes) != 1:
        raise ValueError("All dataset variables must have the same number of samples.")
    num_samples = sample_sizes.pop()

    variable_names = list(dataset.data.keys())
    variable_name_set = set(variable_names)
    num_variables = len(variable_names)

    variable_types: dict[str, str] = {}
    variable_specs: dict[str, dict[str, Any]] = {}
    transform_counter: Counter[str] = Counter()
    parent_map: dict[str, list[str]] = {}

    for name, variable in dataset.scm._variables.items():
        parent_names = list(variable.parent_names)
        parent_map[name] = parent_names
        if isinstance(variable, LinearVariable):
            variable_types[name] = "linear"
            variable_specs[name] = {
                "class": type(variable).__name__,
                "sigma": float(variable.sigma),
                "parents": parent_names,
                "intercept": float(variable._intercept),
                "coefs": dict(variable._coefs),
                "transforms": [],
            }
        elif isinstance(variable, FunctionalVariable):
            transforms = list(variable._transforms or [])
            transform_counter.update(transforms)
            variable_types[name] = "functional"
            variable_specs[name] = {
                "class": type(variable).__name__,
                "sigma": float(variable.sigma),
                "parents": parent_names,
                "intercept": float(variable._intercept or 0.0),
                "coefs": dict(variable._coefs or {}),
                "transforms": transforms,
            }
        else:
            maybe_transforms = getattr(variable, "_transforms", []) or []
            transform_counter.update(str(t) for t in maybe_transforms)
            variable_types[name] = (
                type(variable).__name__.replace("Variable", "").lower()
            )
            variable_specs[name] = {
                "class": type(variable).__name__,
                "sigma": float(variable.sigma)
                if getattr(variable, "sigma", None) is not None
                else None,
                "parents": parent_names,
                "transforms": [str(t) for t in maybe_transforms],
                "extra": _extract_known_variable_extras(variable),
            }

    in_degree = {name: len(parent_map[name]) for name in variable_names}
    out_degree = {name: 0 for name in variable_names}
    for child, parents in parent_map.items():
        for parent in parents:
            if parent in out_degree:
                out_degree[parent] += 1
    num_edges = sum(in_degree.values())

    gaussian_variables = [
        name
        for name, variable in dataset.scm._variables.items()
        if isinstance(variable, GaussianVariable)
    ]
    gaussian_variable_set = set(gaussian_variables)

    treatment_counter = Counter(query.treatment_name for query in dataset.queries)
    outcome_counter = Counter(query.outcome_name for query in dataset.queries)
    pair_counter = Counter(
        f"{query.treatment_name}->{query.outcome_name}" for query in dataset.queries
    )
    observed_counter = Counter(
        observed_name
        for query in dataset.queries
        for observed_name in query.observed_names
    )

    observed_variable_coverage = _safe_ratio(
        len(set(observed_counter) & variable_name_set), num_variables
    )
    treatment_variable_coverage = _safe_ratio(
        len(set(treatment_counter) & variable_name_set), num_variables
    )
    outcome_variable_coverage = _safe_ratio(
        len(set(outcome_counter) & variable_name_set), num_variables
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
    query_density_over_pairs = _safe_ratio(len(dataset.queries), possible_pairs)
    observed_sizes = [len(query.observed_names) for query in dataset.queries]

    expected_keys = {
        (query.treatment_name, query.outcome_name, tuple(query.observed_names))
        for query in dataset.queries
    }
    effect_keys = set(dataset.causal_effects.keys())
    bias_keys = set(dataset.causal_biases.keys())

    data_distribution = {
        name: _distribution_summary(values) for name, values in dataset.data.items()
    }
    pairwise_relations = _pairwise_relation_summary(
        dataset.data, max_pairs=max_pairwise_pairs
    )
    effect_distribution = {
        str(_serialize_key(key)): _distribution_summary(values)
        for key, values in dataset.causal_effects.items()
    }
    bias_distribution = {
        str(_serialize_key(key)): _distribution_summary(values)
        for key, values in dataset.causal_biases.items()
    }
    effect_bias_alignment = {
        str(_serialize_key(key)): _effect_bias_alignment(
            dataset.causal_effects[key], dataset.causal_biases[key]
        )
        for key in sorted(set(dataset.causal_effects) & set(dataset.causal_biases))
    }

    return DatasetCoverageReport(
        num_samples=num_samples,
        num_variables=num_variables,
        num_queries=len(dataset.queries),
        num_edges=num_edges,
        variable_types=variable_types,
        variable_specs=variable_specs,
        parent_map=parent_map,
        in_degree=in_degree,
        out_degree=out_degree,
        transform_usage=dict(transform_counter),
        observed_variable_coverage=observed_variable_coverage,
        treatment_variable_coverage=treatment_variable_coverage,
        outcome_variable_coverage=outcome_variable_coverage,
        treatment_outcome_pair_coverage=treatment_outcome_pair_coverage,
        query_density_over_pairs=query_density_over_pairs,
        observed_size_summary=_summary_from_ints(observed_sizes),
        data_distribution=data_distribution,
        pairwise_relations=pairwise_relations,
        effect_distribution=effect_distribution,
        bias_distribution=bias_distribution,
        effect_bias_alignment=effect_bias_alignment,
        treatment_counts=dict(treatment_counter),
        outcome_counts=dict(outcome_counter),
        observed_counts=dict(observed_counter),
        treatment_outcome_pair_counts=dict(pair_counter),
        missing_effect_keys=sorted(expected_keys - effect_keys),
        missing_bias_keys=sorted(expected_keys - bias_keys),
        extra_effect_keys=sorted(effect_keys - expected_keys),
        extra_bias_keys=sorted(bias_keys - expected_keys),
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


def analyze_scm_dataset_collection(
    collection: SCMDatasetCollection,
) -> DatasetCollectionCoverageReport:
    reports = [analyze_scm_dataset(dataset) for dataset in collection.datasets]
    if not reports:
        raise ValueError("Dataset collection is empty.")

    aggregate_summary = {
        "num_datasets": float(len(reports)),
        "mean_samples": _mean([report.num_samples for report in reports]),
        "mean_variables": _mean([report.num_variables for report in reports]),
        "mean_edges": _mean([report.num_edges for report in reports]),
        "mean_queries": _mean([report.num_queries for report in reports]),
        "mean_observed_variable_coverage": _mean(
            [report.observed_variable_coverage for report in reports]
        ),
        "mean_treatment_variable_coverage": _mean(
            [report.treatment_variable_coverage for report in reports]
        ),
        "mean_outcome_variable_coverage": _mean(
            [report.outcome_variable_coverage for report in reports]
        ),
        "mean_treatment_outcome_pair_coverage": _mean(
            [report.treatment_outcome_pair_coverage for report in reports]
        ),
        "mean_query_density_over_pairs": _mean(
            [report.query_density_over_pairs for report in reports]
        ),
        "max_missing_effect_keys": float(
            max(len(report.missing_effect_keys) for report in reports)
        ),
        "max_missing_bias_keys": float(
            max(len(report.missing_bias_keys) for report in reports)
        ),
    }

    dataset_metric_summary = {
        "num_samples": _metric_summary(
            [float(report.num_samples) for report in reports]
        ),
        "num_variables": _metric_summary(
            [float(report.num_variables) for report in reports]
        ),
        "num_edges": _metric_summary([float(report.num_edges) for report in reports]),
        "num_queries": _metric_summary(
            [float(report.num_queries) for report in reports]
        ),
        "observed_variable_coverage": _metric_summary(
            [report.observed_variable_coverage for report in reports]
        ),
        "treatment_variable_coverage": _metric_summary(
            [report.treatment_variable_coverage for report in reports]
        ),
        "outcome_variable_coverage": _metric_summary(
            [report.outcome_variable_coverage for report in reports]
        ),
        "treatment_outcome_pair_coverage": _metric_summary(
            [report.treatment_outcome_pair_coverage for report in reports]
        ),
        "query_density_over_pairs": _metric_summary(
            [report.query_density_over_pairs for report in reports]
        ),
    }

    structure_summary = {
        "variable_type_frequency": dict(
            Counter(typ for report in reports for typ in report.variable_types.values())
        ),
        "transform_frequency": dict(
            Counter(
                transform
                for report in reports
                for transform, n in report.transform_usage.items()
                for _ in range(n)
            )
        ),
    }

    domain_coverage_summary = _aggregate_domain_coverage(reports)
    pair_relation_summary = _aggregate_pair_relations(reports)

    return DatasetCollectionCoverageReport(
        num_datasets=len(reports),
        per_dataset_reports=reports,
        aggregate_summary=aggregate_summary,
        dataset_metric_summary=dataset_metric_summary,
        structure_summary=structure_summary,
        domain_coverage_summary=domain_coverage_summary,
        pair_relation_summary=pair_relation_summary,
    )


def compare_real_dataset_to_collection(
    real_dataset: SCMDataset,
    synthetic_collection: SCMDatasetCollection,
    *,
    thresholds: SimilarityThresholds | None = None,
) -> RealDatasetSimilarityReport:
    """Measure whether a real dataset is similar enough to a synthetic collection."""
    if thresholds is None:
        thresholds = SimilarityThresholds()

    if not synthetic_collection.datasets:
        raise ValueError("Synthetic collection is empty.")

    compared_variables = sorted(
        set(real_dataset.data.keys()).intersection(
            set(synthetic_collection.datasets[0].data.keys())
        )
    )
    if not compared_variables:
        raise ValueError(
            "No overlapping variables between real dataset and synthetic collection."
        )

    pooled = _pooled_variable_data(synthetic_collection, compared_variables)
    real_report = analyze_scm_dataset(real_dataset)
    synth_report = analyze_scm_dataset_collection(synthetic_collection)

    variable_marginal_distances: dict[str, float] = {}
    variable_outside_fractions: dict[str, float] = {}
    for name in compared_variables:
        real_vals = real_dataset.data[name].detach().float().cpu()
        syn_vals = pooled[name].detach().float().cpu()
        variable_marginal_distances[name] = _quantile_distance(real_vals, syn_vals)
        variable_outside_fractions[name] = _outside_fraction(real_vals, syn_vals)

    marginal_distance_mean = _mean(list(variable_marginal_distances.values()))
    marginal_distance_max = max(variable_marginal_distances.values())
    mean_outside_fraction = _mean(list(variable_outside_fractions.values()))
    max_outside_fraction = max(variable_outside_fractions.values())

    correlation_distance = _correlation_matrix_distance(
        real_dataset.data,
        pooled,
        compared_variables,
    )

    real_pairs = set(real_report.treatment_outcome_pair_counts.keys())
    synth_pairs = {
        pair
        for rep in synth_report.per_dataset_reports
        for pair in rep.treatment_outcome_pair_counts.keys()
    }
    pair_overlap = _safe_ratio(
        len(real_pairs.intersection(synth_pairs)), len(real_pairs)
    )

    score_parts = [
        _bounded_inverse(marginal_distance_mean, thresholds.max_marginal_distance),
        _bounded_inverse(correlation_distance, thresholds.max_correlation_distance),
        _bounded_inverse(mean_outside_fraction, thresholds.max_mean_outside_fraction),
        min(1.0, _safe_ratio(pair_overlap, thresholds.min_pair_overlap)),
    ]
    overall_score = _mean(score_parts)

    similar_enough = (
        marginal_distance_mean <= thresholds.max_marginal_distance
        and correlation_distance <= thresholds.max_correlation_distance
        and mean_outside_fraction <= thresholds.max_mean_outside_fraction
        and pair_overlap >= thresholds.min_pair_overlap
        and overall_score >= thresholds.min_overall_score
    )

    total_synth_samples = int(
        sum(len(next(iter(ds.data.values()))) for ds in synthetic_collection.datasets)
    )

    return RealDatasetSimilarityReport(
        compared_variables=compared_variables,
        num_real_samples=int(len(next(iter(real_dataset.data.values())))),
        num_synthetic_samples_total=total_synth_samples,
        marginal_distance_mean=float(marginal_distance_mean),
        marginal_distance_max=float(marginal_distance_max),
        correlation_distance=float(correlation_distance),
        mean_outside_fraction=float(mean_outside_fraction),
        max_outside_fraction=float(max_outside_fraction),
        treatment_outcome_pair_overlap=float(pair_overlap),
        overall_score=float(overall_score),
        similar_enough=bool(similar_enough),
        variable_marginal_distances=variable_marginal_distances,
        variable_outside_fractions=variable_outside_fractions,
        thresholds={
            "max_marginal_distance": thresholds.max_marginal_distance,
            "max_correlation_distance": thresholds.max_correlation_distance,
            "max_mean_outside_fraction": thresholds.max_mean_outside_fraction,
            "min_pair_overlap": thresholds.min_pair_overlap,
            "min_overall_score": thresholds.min_overall_score,
        },
    )


def generate_collection_analysis_plots(
    collection: SCMDatasetCollection,
    report: DatasetCollectionCoverageReport,
    output_dir: str | Path,
    *,
    max_scatter_pairs: int = 6,
) -> list[Path]:
    """Generate exhaustive visual diagnostics and return produced file paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 1) Domain coverage over collection.
    domain_items = sorted(report.domain_coverage_summary.items())
    if domain_items:
        fig, ax = plt.subplots(figsize=(8, max(3, len(domain_items) * 0.35)))
        ys = np.arange(len(domain_items))
        mins = [item[1]["global_min"] for item in domain_items]
        maxs = [item[1]["global_max"] for item in domain_items]
        ax.hlines(ys, mins, maxs, color="#2E5BFF", linewidth=2.5)
        ax.scatter(mins, ys, color="#1E88E5", s=24)
        ax.scatter(maxs, ys, color="#43A047", s=24)
        ax.set_yticks(ys)
        ax.set_yticklabels([item[0] for item in domain_items])
        ax.set_xlabel("Value range across all subdatasets")
        ax.set_title("Domain coverage by variable")
        fig.tight_layout()
        p = output / "domain_coverage_ranges.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # 2) Dataset-level metric distributions.
    metric_summary = report.dataset_metric_summary
    metric_names = [
        "num_edges",
        "num_queries",
        "observed_variable_coverage",
        "treatment_outcome_pair_coverage",
        "query_density_over_pairs",
    ]
    means = [
        metric_summary[name]["mean"] for name in metric_names if name in metric_summary
    ]
    stds = [
        metric_summary[name]["std"] for name in metric_names if name in metric_summary
    ]
    labels = [name for name in metric_names if name in metric_summary]
    if labels:
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color="#5C6BC0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title("Collection-level metric means ± std across subdatasets")
        fig.tight_layout()
        p = output / "collection_metric_summary.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # 3) Aggregated pairwise relation heatmaps (mean abs corr and stability).
    variable_names = sorted(report.domain_coverage_summary.keys())
    if len(variable_names) >= 2:
        mean_abs = np.zeros((len(variable_names), len(variable_names)), dtype=float)
        std_abs = np.zeros((len(variable_names), len(variable_names)), dtype=float)
        np.fill_diagonal(mean_abs, 1.0)
        np.fill_diagonal(std_abs, 0.0)
        for key, stats in report.pair_relation_summary.items():
            left, right = _parse_pair_key(key)
            if left not in variable_names or right not in variable_names:
                continue
            i = variable_names.index(left)
            j = variable_names.index(right)
            mean_abs[i, j] = mean_abs[j, i] = stats.get("mean_abs_corr", 0.0)
            std_abs[i, j] = std_abs[j, i] = stats.get("std_abs_corr", 0.0)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(mean_abs, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(variable_names)))
        ax.set_yticks(range(len(variable_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha="right")
        ax.set_yticklabels(variable_names)
        ax.set_title("Collection relation strength: mean |corr|")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = output / "collection_pairwise_mean_abs_corr_heatmap.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        im = ax.imshow(std_abs, cmap="viridis", vmin=0.0)
        ax.set_xticks(range(len(variable_names)))
        ax.set_yticks(range(len(variable_names)))
        ax.set_xticklabels(variable_names, rotation=45, ha="right")
        ax.set_yticklabels(variable_names)
        ax.set_title("Collection relation stability: std |corr|")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = output / "collection_pairwise_std_abs_corr_heatmap.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # 4) Top pooled 2D relation plots over the full collection.
    top_pairs = sorted(
        report.pair_relation_summary.items(),
        key=lambda kv: kv[1].get("mean_abs_corr", 0.0),
        reverse=True,
    )[:max_scatter_pairs]
    if top_pairs:
        ncols = 3
        nrows = int(np.ceil(len(top_pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows))
        axes_arr = np.atleast_1d(axes).ravel()
        for ax, (pair_key, stats) in zip(axes_arr, top_pairs):
            left, right = _parse_pair_key(pair_key)
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for ds in collection.datasets:
                if left in ds.data and right in ds.data:
                    xs.append(ds.data[left].detach().cpu().numpy().astype(float))
                    ys.append(ds.data[right].detach().cpu().numpy().astype(float))
            if xs and ys:
                x_all = np.concatenate(xs)
                y_all = np.concatenate(ys)
                ax.hexbin(x_all, y_all, gridsize=36, cmap="Blues", mincnt=1)
            ax.set_xlabel(left)
            ax.set_ylabel(right)
            ax.set_title(f"mean|corr|={stats.get('mean_abs_corr', 0.0):.3f}")
        for ax in axes_arr[len(top_pairs) :]:
            ax.axis("off")
        fig.suptitle("Top pooled 2D relations across collection")
        fig.tight_layout()
        p = output / "collection_top_2d_relations_pooled.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # 5) treatment-outcome pair coverage bars.
    pair_counter: Counter[str] = Counter()
    for r in report.per_dataset_reports:
        pair_counter.update(r.treatment_outcome_pair_counts)
    if pair_counter:
        items = pair_counter.most_common(16)
        fig, ax = plt.subplots(figsize=(max(8, len(items) * 0.5), 4))
        ax.bar([k for k, _ in items], [v for _, v in items], color="#7E57C2")
        ax.set_ylabel("Count")
        ax.set_title("Treatment→Outcome pair frequency (collection)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = output / "treatment_outcome_pair_frequency.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    # 6) transform usage.
    transform_freq = report.structure_summary.get("transform_frequency", {})
    if isinstance(transform_freq, dict) and transform_freq:
        items = sorted(transform_freq.items(), key=lambda kv: kv[1], reverse=True)
        fig, ax = plt.subplots(figsize=(max(7, len(items) * 0.45), 4))
        ax.bar([k for k, _ in items], [v for _, v in items], color="#26A69A")
        ax.set_ylabel("Count")
        ax.set_title("Transformation usage in SCM collection")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = output / "transform_usage.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    return paths


def _extract_known_variable_extras(variable: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in [
        "_num_categories",
        "_temperature",
        "_bias",
        "_parent_kinds",
        "_gaussian_parent_weights",
        "_categorical_parent_weights",
    ]:
        if hasattr(variable, attr):
            value = getattr(variable, attr)
            if isinstance(value, (int, float, str, list, dict)):
                out[attr.removeprefix("_")] = value
    return out


def _aggregate_domain_coverage(
    reports: list[DatasetCoverageReport],
) -> dict[str, dict[str, float]]:
    merged: dict[str, list[dict[str, float]]] = {}
    for report in reports:
        for name, stats in report.data_distribution.items():
            merged.setdefault(name, []).append(stats)
    out: dict[str, dict[str, float]] = {}
    for name, stats_list in merged.items():
        out[name] = {
            "global_min": min(s["min"] for s in stats_list),
            "global_max": max(s["max"] for s in stats_list),
            "mean_mean": _mean([s["mean"] for s in stats_list]),
            "mean_std": _mean([s["std"] for s in stats_list]),
            "mean_p05": _mean([s["p05"] for s in stats_list]),
            "mean_p95": _mean([s["p95"] for s in stats_list]),
        }
    return out


def _aggregate_pair_relations(
    reports: list[DatasetCoverageReport],
) -> dict[str, dict[str, float]]:
    merged: dict[str, list[dict[str, float]]] = {}
    for report in reports:
        for key, stats in report.pairwise_relations.items():
            merged.setdefault(key, []).append(stats)
    out: dict[str, dict[str, float]] = {}
    for key, vals in merged.items():
        abs_corrs = [v["abs_corr"] for v in vals]
        corrs = [v["corr"] for v in vals]
        out[key] = {
            "mean_abs_corr": _mean(abs_corrs),
            "std_abs_corr": _std(abs_corrs),
            "mean_corr": _mean(corrs),
            "std_corr": _std(corrs),
            "max_abs_corr": max(abs_corrs),
            "min_abs_corr": min(abs_corrs),
            "support_count": float(len(vals)),
            "support_fraction": _safe_ratio(len(vals), len(reports)),
        }
    return out


def _pairwise_relation_summary(
    data: dict[str, torch.Tensor], *, max_pairs: int
) -> dict[str, dict[str, float]]:
    names = list(data.keys())
    rows: list[tuple[str, str, float, float]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            x = data[left].detach().float()
            y = data[right].detach().float()
            if x.numel() == 0 or y.numel() == 0:
                continue
            if (
                float(x.std(unbiased=False).item()) < 1e-12
                or float(y.std(unbiased=False).item()) < 1e-12
            ):
                corr = 0.0
            else:
                corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
                if np.isnan(corr):
                    corr = 0.0
            rows.append((left, right, corr, abs(corr)))

    rows.sort(key=lambda t: t[3], reverse=True)
    rows = rows[:max_pairs]
    return {
        f"{left}<->{right}": {"corr": corr, "abs_corr": abs_corr}
        for left, right, corr, abs_corr in rows
    }


def _effect_bias_alignment(
    effect: torch.Tensor, bias: torch.Tensor
) -> dict[str, float]:
    e = effect.detach().float()
    b = bias.detach().float()
    if e.numel() == 0:
        return {"corr": 0.0, "mean_abs_effect": 0.0, "mean_abs_bias": 0.0}
    if (
        float(e.std(unbiased=False).item()) < 1e-12
        or float(b.std(unbiased=False).item()) < 1e-12
    ):
        corr = 0.0
    else:
        corr = float(torch.corrcoef(torch.stack([e, b]))[0, 1].item())
        if np.isnan(corr):
            corr = 0.0
    return {
        "corr": corr,
        "mean_abs_effect": float(e.abs().mean().item()),
        "mean_abs_bias": float(b.abs().mean().item()),
    }


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    x = values.float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "p01": float(torch.quantile(x, 0.01).item()),
        "p05": float(torch.quantile(x, 0.05).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "p99": float(torch.quantile(x, 0.99).item()),
    }


def _nan_fraction(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.isnan(values).sum().item()) / float(values.numel())


def _summary_from_ints(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(values)), "max": float(max(values)), "mean": _mean(values)}


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(float(v) for v in values) / len(values))


def _std(values: list[float | int]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.array([float(v) for v in values], dtype=float)
    return float(arr.std(ddof=0))


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def _parse_pair_key(key: str) -> tuple[str, str]:
    if "<->" not in key:
        return key, key
    left, right = key.split("<->", 1)
    return left, right


def _pooled_variable_data(
    collection: SCMDatasetCollection,
    variable_names: list[str],
) -> dict[str, torch.Tensor]:
    pooled: dict[str, torch.Tensor] = {}
    for name in variable_names:
        chunks = [
            ds.data[name].detach().float().cpu()
            for ds in collection.datasets
            if name in ds.data
        ]
        if not chunks:
            continue
        pooled[name] = torch.cat(chunks)
    return pooled


def _quantile_distance(
    real: torch.Tensor, synthetic: torch.Tensor, *, num_q: int = 21
) -> float:
    if real.numel() == 0 or synthetic.numel() == 0:
        return 1.0
    qs = torch.linspace(0.0, 1.0, num_q)
    rq = torch.quantile(real, qs)
    sq = torch.quantile(synthetic, qs)
    scale = float(torch.quantile(synthetic, 0.95) - torch.quantile(synthetic, 0.05))
    if scale <= 1e-12:
        scale = float(torch.std(synthetic, unbiased=False).item()) + 1e-6
    return float(torch.mean(torch.abs(rq - sq)).item() / max(scale, 1e-6))


def _outside_fraction(real: torch.Tensor, synthetic: torch.Tensor) -> float:
    if real.numel() == 0 or synthetic.numel() == 0:
        return 1.0
    lo = torch.quantile(synthetic, 0.01)
    hi = torch.quantile(synthetic, 0.99)
    outside = (real < lo) | (real > hi)
    return float(outside.float().mean().item())


def _correlation_matrix_distance(
    real_data: dict[str, torch.Tensor],
    synthetic_data: dict[str, torch.Tensor],
    variable_names: list[str],
) -> float:
    if len(variable_names) < 2:
        return 0.0
    r = np.stack(
        [
            real_data[name].detach().cpu().numpy().astype(float)
            for name in variable_names
        ],
        axis=0,
    )
    s = np.stack(
        [
            synthetic_data[name].detach().cpu().numpy().astype(float)
            for name in variable_names
        ],
        axis=0,
    )
    corr_r = np.nan_to_num(np.corrcoef(r), nan=0.0)
    corr_s = np.nan_to_num(np.corrcoef(s), nan=0.0)
    diff = np.abs(corr_r - corr_s)
    triu = np.triu_indices(diff.shape[0], 1)
    if len(triu[0]) == 0:
        return 0.0
    return float(diff[triu].mean())


def _bounded_inverse(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return float(max(0.0, 1.0 - (value / threshold)))


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _serialize_key(key: tuple[str, str, tuple[str, ...]]) -> tuple[str, str, list[str]]:
    return key[0], key[1], list(key[2])
