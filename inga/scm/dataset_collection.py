"""Dataset-of-datasets utilities for random SCM generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from inga.scm.dataset import (
    SCMDataset,
    SCMDatasetConfig,
    generate_scm_dataset,
    load_scm_dataset,
)
from inga.scm.random import RandomSCMConfig


@dataclass(frozen=True)
class SCMDatasetCollection:
    """A collection of SCM datasets, each sampled from its own random SCM."""

    datasets: list[SCMDataset]

    def save(self, path: str | Path) -> None:
        """Save collection to ``path`` directory.

        The directory will contain:
        - ``manifest.json``
        - per-subdataset files ``dataset_XXXX.json/.pt``
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, str | int]] = []
        for idx, dataset in enumerate(self.datasets):
            stem = f"dataset_{idx:04d}"
            dataset.save(root / stem)
            entries.append({"index": idx, "stem": stem})

        manifest = {
            "format": "scm_dataset_collection_v1",
            "num_datasets": len(self.datasets),
            "datasets": entries,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))


@dataclass(frozen=True)
class SCMDatasetCollectionConfig:
    """Configuration for generating a collection of random SCM datasets."""

    scm_config: RandomSCMConfig
    num_datasets: int = 10
    num_samples: int = 1000
    num_queries: int = 10
    min_observed: int = 1
    seed: int | None = None
    treatment_name: str | None = None


def generate_scm_dataset_collection(
    config: SCMDatasetCollectionConfig,
) -> SCMDatasetCollection:
    """Generate multiple SCM datasets, each from a distinct random SCM."""
    if config.num_datasets <= 0:
        raise ValueError("num_datasets must be positive.")

    base_seed = (
        config.seed
        if config.seed is not None
        else (config.scm_config.seed if config.scm_config.seed is not None else 0)
    )

    datasets: list[SCMDataset] = []
    for idx in range(config.num_datasets):
        scm_seed = base_seed + idx
        dataset_seed = base_seed + 100_000 + idx

        scm_config = replace(config.scm_config, seed=scm_seed)
        ds_config = SCMDatasetConfig(
            scm_config=scm_config,
            num_samples=config.num_samples,
            num_queries=config.num_queries,
            min_observed=config.min_observed,
            seed=dataset_seed,
            treatment_name=config.treatment_name,
        )
        datasets.append(generate_scm_dataset(ds_config))

    return SCMDatasetCollection(datasets=datasets)


def load_scm_dataset_collection(path: str | Path) -> SCMDatasetCollection:
    """Load a saved dataset collection from directory ``path``."""
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Collection manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Invalid collection manifest: 'datasets' must be a list.")

    datasets: list[SCMDataset] = []
    for entry in entries:
        if not isinstance(entry, dict) or "stem" not in entry:
            raise ValueError(
                "Invalid collection manifest entry; expected a dict with 'stem'."
            )
        stem = str(entry["stem"])
        datasets.append(load_scm_dataset(root / stem))

    return SCMDatasetCollection(datasets=datasets)
