"""Compare a real SCM dataset against a synthetic collection and decide similarity."""

from __future__ import annotations

import argparse
import json

from inga.scm import (
    SimilarityThresholds,
    compare_real_dataset_to_collection,
    load_scm_dataset,
    load_scm_dataset_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        type=str,
        default="datasets/scm_dataset_example",
        help="Real dataset base path (without .json/.pt).",
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        default="datasets/random_scm_collection",
        help="Synthetic collection directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="datasets/real_vs_synth_similarity_report.json",
        help="Path for similarity report JSON.",
    )
    parser.add_argument("--max-marginal-distance", type=float, default=0.50)
    parser.add_argument("--max-correlation-distance", type=float, default=0.35)
    parser.add_argument("--max-mean-outside-fraction", type=float, default=0.15)
    parser.add_argument("--min-pair-overlap", type=float, default=0.10)
    parser.add_argument("--min-overall-score", type=float, default=0.55)
    args = parser.parse_args()

    real = load_scm_dataset(args.real)
    synth = load_scm_dataset_collection(args.synthetic)
    thresholds = SimilarityThresholds(
        max_marginal_distance=args.max_marginal_distance,
        max_correlation_distance=args.max_correlation_distance,
        max_mean_outside_fraction=args.max_mean_outside_fraction,
        min_pair_overlap=args.min_pair_overlap,
        min_overall_score=args.min_overall_score,
    )
    report = compare_real_dataset_to_collection(real, synth, thresholds=thresholds)

    print("Real vs synthetic collection similarity")
    print(f"Real dataset: {args.real}")
    print(f"Synthetic collection: {args.synthetic}")
    print(f"Compared variables: {len(report.compared_variables)}")
    print(f"Overall score: {report.overall_score:.3f}")
    print(
        f"Marginal distance (mean/max): {report.marginal_distance_mean:.3f}/{report.marginal_distance_max:.3f}"
    )
    print(f"Correlation distance: {report.correlation_distance:.3f}")
    print(
        f"Outside fraction (mean/max): {report.mean_outside_fraction:.3f}/{report.max_outside_fraction:.3f}"
    )
    print(
        f"Treatment-outcome pair overlap: {report.treatment_outcome_pair_overlap:.3f}"
    )
    print(f"Similar enough: {report.similar_enough}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved similarity report to: {args.output_json}")


if __name__ == "__main__":
    main()
