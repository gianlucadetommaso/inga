"""Analyze an SCM dataset and print coverage diagnostics.

Usage:
    uv run python -m examples.analyze_scm_dataset --input datasets/random_scm_dataset
"""

from __future__ import annotations

import argparse
import json

from inga.scm import analyze_scm_dataset, load_scm_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="datasets/random_scm_dataset",
        help="Dataset base path (reads <input>.json and <input>.pt).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full JSON report.",
    )
    args = parser.parse_args()

    dataset = load_scm_dataset(args.input)
    report = analyze_scm_dataset(dataset)
    report_dict = report.to_dict()

    print("SCM dataset analysis")
    print(f"Samples: {report.num_samples}")
    print(f"Variables: {report.num_variables}")
    print(f"Queries: {report.num_queries}")
    print(
        "Coverage (observed/treatment/outcome/pairs): "
        f"{report.observed_variable_coverage:.3f}/"
        f"{report.treatment_variable_coverage:.3f}/"
        f"{report.outcome_variable_coverage:.3f}/"
        f"{report.treatment_outcome_pair_coverage:.3f}"
    )
    print(
        "Missing keys (effects/biases): "
        f"{len(report.missing_effect_keys)}/{len(report.missing_bias_keys)}"
    )

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Saved analysis JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
