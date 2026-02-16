"""Structural Causal Model module."""

from inga.scm.base import SCM
from inga.scm.causal_effect import CausalEffectMixin
from inga.scm.html import HTMLMixin
from inga.scm.random import RandomSCMConfig, random_scm
from inga.scm.dataset import (
    CausalQueryConfig,
    SCMDataset,
    SCMDatasetConfig,
    generate_scm_dataset,
    load_scm_dataset,
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
    "generate_scm_dataset",
    "load_scm_dataset",
]
