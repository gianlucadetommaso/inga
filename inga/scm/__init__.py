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
from inga.scm.variable import GaussianVariable, LinearVariable, Variable

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
    "Variable",
    "GaussianVariable",
    "LinearVariable",
]
