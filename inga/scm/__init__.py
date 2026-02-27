"""Structural Causal Model module."""

from inga.scm.base import SCM
from inga.scm.causal_effect import CausalEffectMixin
from inga.scm.html import HTMLMixin
from inga.scm.random import RandomSCMConfig, random_scm
from inga.scm.dataset import (
    CausalQueryConfig,
    SCMDataset,
    SCMDatasetConfig,
    generate_dataset_from_scm,
    generate_scm_dataset,
    load_scm_dataset,
)
from inga.scm.dataset_analysis import (
    DatasetCollectionCoverageReport,
    DatasetCoverageReport,
    RealDatasetSimilarityReport,
    SimilarityThresholds,
    analyze_scm_dataset,
    analyze_scm_dataset_collection,
    compare_real_dataset_to_collection,
    infer_similarity_key_factors,
    generate_collection_analysis_plots,
)
from inga.scm.dataset_collection import (
    SCMDatasetCollection,
    SCMDatasetCollectionConfig,
    generate_scm_dataset_collection,
    load_scm_dataset_collection,
)
from inga.scm.variable import (
    CategoricalVariable,
    GaussianVariable,
    LinearVariable,
    Variable,
)

__all__ = [
    "SCM",
    "CausalEffectMixin",
    "HTMLMixin",
    "RandomSCMConfig",
    "random_scm",
    "CausalQueryConfig",
    "SCMDataset",
    "SCMDatasetConfig",
    "generate_dataset_from_scm",
    "generate_scm_dataset",
    "load_scm_dataset",
    "SCMDatasetCollection",
    "SCMDatasetCollectionConfig",
    "generate_scm_dataset_collection",
    "load_scm_dataset_collection",
    "DatasetCoverageReport",
    "DatasetCollectionCoverageReport",
    "SimilarityThresholds",
    "RealDatasetSimilarityReport",
    "analyze_scm_dataset",
    "analyze_scm_dataset_collection",
    "compare_real_dataset_to_collection",
    "infer_similarity_key_factors",
    "generate_collection_analysis_plots",
    "Variable",
    "GaussianVariable",
    "CategoricalVariable",
    "LinearVariable",
]
