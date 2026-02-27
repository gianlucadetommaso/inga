"""Tests for SCM dataset analysis utilities."""

from __future__ import annotations

from inga.scm import SCMDatasetConfig, analyze_scm_dataset, generate_scm_dataset
from inga.scm.random import RandomSCMConfig


def test_analyze_scm_dataset_reports_basic_coverage() -> None:
    """Analysis should produce sane coverage and consistency fields."""
    dataset = generate_scm_dataset(
        SCMDatasetConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.5, seed=7),
            num_samples=32,
            num_queries=3,
            min_observed=1,
            seed=11,
        )
    )

    report = analyze_scm_dataset(dataset)

    assert report.num_samples == 32
    assert report.num_variables == 5
    assert report.num_queries == len(dataset.queries)
    assert 0.0 <= report.observed_variable_coverage <= 1.0
    assert 0.0 <= report.treatment_variable_coverage <= 1.0
    assert 0.0 <= report.outcome_variable_coverage <= 1.0
    assert 0.0 <= report.treatment_outcome_pair_coverage <= 1.0
    assert report.missing_effect_keys == []
    assert report.missing_bias_keys == []
    assert report.extra_effect_keys == []
    assert report.extra_bias_keys == []

    report_dict = report.to_dict()
    assert report_dict["num_samples"] == 32
    assert "coverage" in report_dict
    assert "counts" in report_dict
    assert "nan_fraction" in report_dict
