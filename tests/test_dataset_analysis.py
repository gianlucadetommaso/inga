"""Tests for SCM dataset analysis utilities."""

from __future__ import annotations

from inga.scm import (
    SCMDatasetCollectionConfig,
    SCMDatasetConfig,
    analyze_scm_dataset,
    analyze_scm_dataset_collection,
    generate_collection_analysis_plots,
    generate_scm_dataset,
    generate_scm_dataset_collection,
    load_scm_dataset_collection,
)
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
    assert "observed_size_summary" in report_dict
    assert "parent_map" in report_dict
    assert "transform_usage" in report_dict
    assert "pairwise_relations" in report_dict
    assert "effect_bias_alignment" in report_dict


def test_dataset_collection_generation_save_load_and_analysis(tmp_path) -> None:
    """Collection generation should support save/load and aggregate analysis."""
    collection = generate_scm_dataset_collection(
        SCMDatasetCollectionConfig(
            scm_config=RandomSCMConfig(num_variables=4, parent_prob=0.5, seed=17),
            num_datasets=3,
            num_samples=24,
            num_queries=2,
            min_observed=1,
            seed=101,
        )
    )
    assert len(collection.datasets) == 3

    target = tmp_path / "scm_collection"
    collection.save(target)
    loaded = load_scm_dataset_collection(target)
    assert len(loaded.datasets) == 3

    report = analyze_scm_dataset_collection(loaded)
    assert report.num_datasets == 3
    assert len(report.per_dataset_reports) == 3
    assert report.aggregate_summary["mean_samples"] == 24.0
    assert report.aggregate_summary["mean_queries"] >= 2.0
    assert report.aggregate_summary["mean_edges"] >= 0.0
    assert "num_edges" in report.dataset_metric_summary
    assert "variable_type_frequency" in report.structure_summary
    assert "domain_coverage_summary" in report.to_dict()

    report_dict = report.to_dict()
    assert report_dict["num_datasets"] == 3
    assert "aggregate_summary" in report_dict
    assert len(report_dict["per_dataset_reports"]) == 3


def test_generate_collection_analysis_plots(tmp_path) -> None:
    """Plot generation should produce expected diagnostic image files."""
    collection = generate_scm_dataset_collection(
        SCMDatasetCollectionConfig(
            scm_config=RandomSCMConfig(num_variables=5, parent_prob=0.5, seed=23),
            num_datasets=2,
            num_samples=20,
            num_queries=2,
            min_observed=1,
            seed=55,
        )
    )
    report = analyze_scm_dataset_collection(collection)
    output_dir = tmp_path / "analysis_plots"

    paths = generate_collection_analysis_plots(collection, report, output_dir)

    assert len(paths) >= 3
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
