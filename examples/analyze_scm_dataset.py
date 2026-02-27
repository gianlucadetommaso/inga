"""Analyze an SCM dataset collection and print detailed diagnostics.

Usage:
    uv run python -m examples.analyze_scm_dataset --input datasets/random_scm_collection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inga.scm import (
    SCMDatasetCollection,
    analyze_scm_dataset_collection,
    generate_collection_analysis_plots,
    load_scm_dataset,
    load_scm_dataset_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/random_scm_collection",
        help="Collection directory (expects manifest + per-subdataset files).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full JSON report.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="Optional directory where exhaustive analysis plots are saved.",
    )
    parser.add_argument(
        "--max-scatter-pairs",
        type=int,
        default=6,
        help="Max number of top-relationship 2D scatter pairs to plot.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if (input_path / "manifest.json").exists():
        collection = load_scm_dataset_collection(input_path)
    elif input_path.exists() and input_path.is_dir():
        raise FileNotFoundError(
            f"Input directory exists but manifest.json is missing: {input_path}. "
            "If dataset generation is still running, wait for completion and retry."
        )
    else:
        # Backward-compatible path: allow analyzing a single dataset base path.
        single_dataset = load_scm_dataset(input_path)
        collection = SCMDatasetCollection(datasets=[single_dataset])

    report = analyze_scm_dataset_collection(collection)
    report_dict = report.to_dict()

    print("SCM dataset collection analysis")
    print(f"Subdatasets: {report.num_datasets}")
    aggregate = report.aggregate_summary
    print(
        "Aggregate means (samples/variables/queries): "
        f"{aggregate['mean_samples']:.1f}/"
        f"{aggregate['mean_variables']:.1f}/"
        f"{aggregate['mean_queries']:.1f}"
    )
    print(
        "Aggregate coverage (observed/treatment/outcome/pairs): "
        f"{aggregate['mean_observed_variable_coverage']:.3f}/"
        f"{aggregate['mean_treatment_variable_coverage']:.3f}/"
        f"{aggregate['mean_outcome_variable_coverage']:.3f}/"
        f"{aggregate['mean_treatment_outcome_pair_coverage']:.3f}"
    )
    print(
        "Aggregate query density over Gaussian pairs: "
        f"{aggregate['mean_query_density_over_pairs']:.3f}"
    )
    print(
        "Aggregate structure (variables/edges): "
        f"{aggregate['mean_variables']:.2f}/{aggregate['mean_edges']:.2f}"
    )
    print(
        "Worst missing keys (effects/biases): "
        f"{int(aggregate['max_missing_effect_keys'])}/"
        f"{int(aggregate['max_missing_bias_keys'])}"
    )

    transform_frequency = report.structure_summary.get("transform_frequency", {})
    if isinstance(transform_frequency, dict) and transform_frequency:
        top_transforms = sorted(
            transform_frequency.items(), key=lambda kv: kv[1], reverse=True
        )[:8]
        print(
            "Top transforms: "
            + ", ".join(f"{name}:{count}" for name, count in top_transforms)
        )

    metric_summary = report.dataset_metric_summary
    if "num_edges" in metric_summary and "num_queries" in metric_summary:
        print(
            "Dataset spread edges/queries (p25-p50-p75): "
            f"{metric_summary['num_edges']['p25']:.1f}-"
            f"{metric_summary['num_edges']['p50']:.1f}-"
            f"{metric_summary['num_edges']['p75']:.1f} / "
            f"{metric_summary['num_queries']['p25']:.1f}-"
            f"{metric_summary['num_queries']['p50']:.1f}-"
            f"{metric_summary['num_queries']['p75']:.1f}"
        )

    print(
        "Coverage spread (pair coverage p25-p50-p75): "
        f"{metric_summary['treatment_outcome_pair_coverage']['p25']:.3f}-"
        f"{metric_summary['treatment_outcome_pair_coverage']['p50']:.3f}-"
        f"{metric_summary['treatment_outcome_pair_coverage']['p75']:.3f}"
    )

    strongest_pairs = sorted(
        report.pair_relation_summary.items(),
        key=lambda kv: kv[1].get("mean_abs_corr", 0.0),
        reverse=True,
    )[:5]
    if strongest_pairs:
        print(
            "Top collection relations (mean|corr|): "
            + ", ".join(
                f"{pair}:{stats.get('mean_abs_corr', 0.0):.3f}"
                for pair, stats in strongest_pairs
            )
        )

    widest_domains = sorted(
        report.domain_coverage_summary.items(),
        key=lambda kv: kv[1]["global_max"] - kv[1]["global_min"],
        reverse=True,
    )[:5]
    if widest_domains:
        print(
            "Widest domains (min,max): "
            + ", ".join(
                f"{name}:[{stats['global_min']:.2f},{stats['global_max']:.2f}]"
                for name, stats in widest_domains
            )
        )

    print(
        "Collection-level report mode: no per-subdataset plots are produced by default."
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Saved analysis JSON to: {args.output_json}")

    if args.plots_dir:
        paths = generate_collection_analysis_plots(
            collection,
            report,
            args.plots_dir,
            max_scatter_pairs=max(1, args.max_scatter_pairs),
        )
        print(f"Saved {len(paths)} analysis plot(s) to: {args.plots_dir}")


if __name__ == "__main__":
    main()
