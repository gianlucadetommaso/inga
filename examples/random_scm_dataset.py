"""Generate and save a random SCM dataset.

Usage:
    uv run python -m examples.random_scm_dataset
"""

from __future__ import annotations

import argparse

from inga.scm.dataset import SCMDatasetConfig, generate_scm_dataset
from inga.scm.random import RandomSCMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/random_scm_dataset",
        help="Output base path (writes <output>.json and <output>.pt).",
    )
    parser.add_argument(
        "--num-variables",
        type=int,
        default=6,
        help="Number of variables in the random SCM.",
    )
    parser.add_argument(
        "--parent-prob",
        type=float,
        default=0.4,
        help="Probability of creating an edge from an earlier variable.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of generated samples.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=4,
        help="Number of sampled causal queries.",
    )
    parser.add_argument(
        "--min-observed",
        type=int,
        default=1,
        help="Minimum number of observed variables per query.",
    )
    parser.add_argument(
        "--scm-seed",
        type=int,
        default=7,
        help="Seed used to sample the random SCM graph/parameters.",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Seed used to sample dataset queries.",
    )
    args = parser.parse_args()

    config = SCMDatasetConfig(
        scm_config=RandomSCMConfig(
            num_variables=args.num_variables,
            parent_prob=args.parent_prob,
            seed=args.scm_seed,
        ),
        num_samples=args.samples,
        num_queries=args.queries,
        min_observed=args.min_observed,
        seed=args.dataset_seed,
    )

    dataset = generate_scm_dataset(config)
    dataset.save(args.output)

    print("Generated random SCM dataset")
    print(f"Saved dataset to: {args.output}.json/.pt")
    print(f"Variables: {list(dataset.data.keys())}")
    print(f"Queries: {len(dataset.queries)}")


if __name__ == "__main__":
    main()
