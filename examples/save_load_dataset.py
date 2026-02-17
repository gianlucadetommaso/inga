"""Minimal example: generate, save, and load an SCM dataset.

Usage:
    uv run python -m examples.save_load_dataset
    uv run python -m examples.save_load_dataset --output plots/scm_dataset_example
"""

from __future__ import annotations

import argparse

from inga.scm.dataset import SCMDatasetConfig, generate_scm_dataset, load_scm_dataset
from inga.scm.random import RandomSCMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/scm_dataset_example",
        help="Output base path (writes <output>.json and <output>.pt).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of generated samples.",
    )
    args = parser.parse_args()

    config = SCMDatasetConfig(
        scm_config=RandomSCMConfig(num_variables=4, parent_prob=0.5, seed=7),
        num_samples=args.samples,
        num_queries=2,
        min_observed=1,
        seed=42,
    )

    dataset = generate_scm_dataset(config)
    dataset.save(args.output)
    loaded = load_scm_dataset(args.output)

    print(f"Saved dataset to: {args.output}.json/.pt")
    print(f"Loaded variables: {list(loaded.data.keys())}")
    print(f"Loaded queries: {len(loaded.queries)}")


if __name__ == "__main__":
    main()
