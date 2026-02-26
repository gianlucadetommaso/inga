"""Targeted unit tests for ``inga.scm.dataset_core`` helpers."""

from __future__ import annotations

import random

import pytest
import torch

from inga.scm.base import SCM
from inga.scm.dataset_core import (
    CausalQueryConfig,
    _class_path,
    _query_key,
    _sample_query,
    _serialize_scm,
    generate_dataset_from_scm,
)
from inga.scm.variable.base import Variable
from inga.scm.variable.linear import LinearVariable


class ConstantVariable(Variable):
    def __init__(self, name: str, sigma: float = 1.0) -> None:
        super().__init__(name=name)
        self.sigma = sigma

    def f_mean(self, parents: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.tensor(0.0)


class BadToDictVariable(ConstantVariable):
    def to_dict(self) -> str:
        return "not-a-dict"


def test_sample_query_requires_at_least_two_variables() -> None:
    with pytest.raises(ValueError, match="At least two variables"):
        _sample_query(
            random.Random(0),
            variable_names=["X"],
            queryable_names=["X"],
            min_observed=1,
        )


def test_sample_query_rejects_unknown_treatment() -> None:
    with pytest.raises(ValueError, match="Unknown treatment_name"):
        _sample_query(
            random.Random(0),
            variable_names=["X", "Y"],
            queryable_names=["X", "Y"],
            min_observed=1,
            treatment_name="Z",
        )


def test_sample_query_without_explicit_treatment_uses_observed_treatment() -> None:
    query = _sample_query(
        random.Random(1),
        variable_names=["X", "Y", "Z"],
        queryable_names=["X", "Y", "Z"],
        min_observed=1,
    )

    assert query.treatment_name in query.observed_names
    assert query.outcome_name not in query.observed_names


def test_sample_query_with_treatment_includes_it_in_observed() -> None:
    query = _sample_query(
        random.Random(2),
        variable_names=["X", "Y", "Z"],
        queryable_names=["X", "Y", "Z"],
        min_observed=1,
        treatment_name="X",
    )

    assert query.treatment_name == "X"
    assert "X" in query.observed_names
    assert query.outcome_name != "X"


def test_query_key_shapes_tuple() -> None:
    query = CausalQueryConfig(
        treatment_name="X",
        outcome_name="Y",
        observed_names=["X", "Z"],
    )
    assert _query_key(query) == ("X", "Y", ("X", "Z"))


def test_generate_dataset_from_scm_validates_basic_arguments() -> None:
    scm = SCM(
        variables=[
            LinearVariable(
                name="X", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="Y",
                parent_names=["X"],
                sigma=1.0,
                coefs={"X": 1.0},
                intercept=0.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="num_samples"):
        generate_dataset_from_scm(scm=scm, num_samples=0)

    with pytest.raises(ValueError, match="num_queries"):
        generate_dataset_from_scm(scm=scm, num_samples=2, num_queries=0)

    with pytest.raises(ValueError, match=r"must match len\(queries\)"):
        generate_dataset_from_scm(
            scm=scm,
            num_samples=2,
            num_queries=2,
            queries=[
                CausalQueryConfig(
                    treatment_name="X",
                    outcome_name="Y",
                    observed_names=["X"],
                )
            ],
        )


def test_class_path_returns_fully_qualified_name() -> None:
    assert _class_path(ConstantVariable).endswith(".ConstantVariable")


def test_serialize_scm_rejects_non_dict_payload_from_to_dict() -> None:
    scm = SCM(variables=[BadToDictVariable(name="X", sigma=1.0)])

    with pytest.raises(ValueError, match="Unsupported variable type"):
        _serialize_scm(scm)
