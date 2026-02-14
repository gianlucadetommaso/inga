"""Generate and persist a corpus of synthetic SEM datasets.

This is the first step of the synthetic->real benchmark.

It uses `SEMDataset.save()` so the corpus can be reused across runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".data/synth_corpus",
        help="Where to store the generated corpus (manifest.json + dataset_XXXXX.{json,pt})",
    )
    parser.add_argument("--num-datasets", type=int, default=200)
    parser.add_argument("--num-variables", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--min-observed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format": "steindag-synth-corpus-v1",
        "num_datasets": args.num_datasets,
        "datasets": [],
        "dataset_config": {
            "num_variables": args.num_variables,
            "num_samples": args.num_samples,
            "num_queries": args.num_queries,
            "min_observed": args.min_observed,
            "seed": args.seed,
        },
    }

    for i in range(args.num_datasets):
        ds_seed = args.seed + i
        dataset = generate_sem_dataset(
            SEMDatasetConfig(
                sem_config=RandomSEMConfig(
                    num_variables=args.num_variables,
                    parent_prob=0.6,
                    nonlinear_prob=0.8,
                    sigma_range=(0.7, 1.2),
                    coef_range=(-1.0, 1.0),
                    intercept_range=(-0.5, 0.5),
                    seed=ds_seed,
                ),
                num_samples=args.num_samples,
                num_queries=args.num_queries,
                min_observed=args.min_observed,
                seed=ds_seed,
            )
        )

        rel = f"dataset_{i:05d}"
        dataset.save(out_dir / rel)
        manifest["datasets"].append({"id": i, "seed": ds_seed, "path": rel})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {args.num_datasets} datasets to: {out_dir}")


if __name__ == "__main__":
    main()
