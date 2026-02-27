"""Generate and save a random SCM dataset collection (dataset-of-datasets).

Usage:
    uv run python -m examples.random_scm_collection
"""

from __future__ import annotations

import argparse

from inga.scm import (
    RandomSCMConfig,
    SCMDatasetCollectionConfig,
    generate_scm_dataset_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/random_scm_collection",
        help="Output directory for collection (writes manifest + per-subdataset files).",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=8,
        help="Number of subdatasets (one SCM per subdataset).",
    )
    parser.add_argument(
        "--num-variables",
        type=int,
        default=6,
        help="Number of variables in each random SCM.",
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
        help="Number of generated samples per subdataset.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=4,
        help="Number of sampled causal queries per subdataset.",
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
        help="Seed base for random SCM graph/parameters.",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Seed base for query and sample generation.",
    )
    args = parser.parse_args()

    config = SCMDatasetCollectionConfig(
        scm_config=RandomSCMConfig(
            num_variables=args.num_variables,
            parent_prob=args.parent_prob,
            seed=args.scm_seed,
        ),
        num_datasets=args.num_datasets,
        num_samples=args.samples,
        num_queries=args.queries,
        min_observed=args.min_observed,
        seed=args.dataset_seed,
    )

    collection = generate_scm_dataset_collection(config)
    collection.save(args.output)

    print("Generated random SCM dataset collection")
    print(f"Saved collection to: {args.output}")
    print(f"Subdatasets: {len(collection.datasets)}")
    if collection.datasets:
        print(
            f"Variables per dataset (first): {list(collection.datasets[0].data.keys())}"
        )
        print(f"Queries per dataset (first): {len(collection.datasets[0].queries)}")


if __name__ == "__main__":
    main()
