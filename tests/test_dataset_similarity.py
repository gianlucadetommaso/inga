"""Tests for real-vs-synthetic dataset similarity utilities."""

from __future__ import annotations

from inga.scm import (
    SCMDatasetCollectionConfig,
    SCMDatasetConfig,
    SimilarityThresholds,
    compare_real_dataset_to_collection,
    generate_scm_dataset,
    generate_scm_dataset_collection,
    infer_similarity_key_factors,
)
from inga.scm.random import RandomSCMConfig


def test_compare_real_dataset_to_collection_returns_expected_fields() -> None:
    synthetic = generate_scm_dataset_collection(
        SCMDatasetCollectionConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.45, seed=31),
            num_datasets=3,
            num_samples=64,
            num_queries=2,
            seed=101,
        )
    )
    real = generate_scm_dataset(
        SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.45, seed=32),
            num_samples=64,
            num_queries=2,
            seed=202,
        )
    )

    report = compare_real_dataset_to_collection(real, synthetic)

    assert len(report.compared_variables) > 0
    assert report.num_real_samples == 64
    assert report.num_synthetic_samples_total == 3 * 64
    assert 0.0 <= report.overall_score <= 1.0
    assert report.marginal_distance_mean >= 0.0
    assert report.marginal_distance_max >= report.marginal_distance_mean
    assert report.correlation_distance >= 0.0
    assert 0.0 <= report.mean_outside_fraction <= 1.0
    assert 0.0 <= report.max_outside_fraction <= 1.0
    assert 0.0 <= report.treatment_outcome_pair_overlap <= 1.0
    assert set(report.variable_marginal_distances.keys()) == set(
        report.compared_variables
    )
    assert set(report.variable_outside_fractions.keys()) == set(
        report.compared_variables
    )


def test_similarity_thresholds_can_force_negative_decision() -> None:
    synthetic = generate_scm_dataset_collection(
        SCMDatasetCollectionConfig(
            scm_config=RandomSCMConfig(num_variables=4, parent_prob=0.4, seed=11),
            num_datasets=2,
            num_samples=48,
            num_queries=2,
            seed=77,
        )
    )
    real = generate_scm_dataset(
        SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=4, parent_prob=0.4, seed=12),
            num_samples=48,
            num_queries=2,
            seed=88,
        )
    )

    strict = SimilarityThresholds(
        max_marginal_distance=0.0,
        max_correlation_distance=0.0,
        max_mean_outside_fraction=0.0,
        min_pair_overlap=1.0,
        min_overall_score=1.0,
    )
    report = compare_real_dataset_to_collection(real, synthetic, thresholds=strict)
    assert report.similar_enough is False


def test_infer_similarity_key_factors_returns_ranked_factors() -> None:
    params = [
        {"parent_prob": 0.2, "nonlinear_prob": 0.2},
        {"parent_prob": 0.4, "nonlinear_prob": 0.4},
        {"parent_prob": 0.6, "nonlinear_prob": 0.8},
    ]
    scores = [0.1, 0.5, 0.9]

    ranked = infer_similarity_key_factors(params, scores)

    assert len(ranked) == 2
    factors = [name for name, _ in ranked]
    assert "parent_prob" in factors
    assert "nonlinear_prob" in factors
