"""Tests for Variable and LinearVariable classes."""

import pytest
import torch
from steindag.variable.base import Variable
from steindag.variable.linear import LinearVariable


class ConcreteVariable(Variable):
    """Concrete implementation of Variable for testing."""

    def f_bar(self, parents: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return zero tensor for testing."""
        return torch.tensor(0.0)


class TestVariable:
    """Tests for base Variable class."""

    def test_parent_names_defaults_to_empty_list(self) -> None:
        """Test that parent_names defaults to empty list when not provided."""
        var = ConcreteVariable(name="X", sigma=1.0)

        assert var.parent_names == []

    def test_parent_names_none_becomes_empty_list(self) -> None:
        """Test that parent_names=None becomes empty list."""
        var = ConcreteVariable(name="X", sigma=1.0, parent_names=None)

        assert var.parent_names == []

    def test_parent_names_converted_to_list(self) -> None:
        """Test that parent_names iterable is converted to list."""
        var = ConcreteVariable(name="X", sigma=1.0, parent_names=("A", "B"))

        assert var.parent_names == ["A", "B"]
        assert isinstance(var.parent_names, list)


class TestLinearVariable:
    """Tests for LinearVariable class."""

    def test_init_stores_attributes(self) -> None:
        """Test that __init__ correctly stores all attributes."""
        var = LinearVariable(
            name="X",
            parent_names=["Z"],
            sigma=2.0,
            coefs={"Z": 1.5},
            intercept=0.5,
        )

        assert var.name == "X"
        assert list(var.parent_names) == ["Z"]
        assert var.sigma == 2.0
        assert var._coefs == {"Z": 1.5}
        assert var._intercept == 0.5

    def test_init_no_parents(self) -> None:
        """Test initialization with no parents (root node)."""
        var = LinearVariable(
            name="Z",
            parent_names=[],
            sigma=1.0,
            coefs={},
            intercept=0.0,
        )

        assert var.name == "Z"
        assert list(var.parent_names) == []
        assert var._coefs == {}

    def test_f_bar_no_parents(self) -> None:
        """Test f_bar with no parents returns intercept."""
        var = LinearVariable(
            name="Z",
            parent_names=[],
            sigma=1.0,
            coefs={},
            intercept=3.0,
        )

        result = var.f_bar({})
        assert result == 3.0

    def test_f_bar_one_parent(self) -> None:
        """Test f_bar with one parent computes correct linear combination."""
        var = LinearVariable(
            name="X",
            parent_names=["Z"],
            sigma=1.0,
            coefs={"Z": 2.0},
            intercept=1.0,
        )

        z = torch.tensor([1.0, 2.0, 3.0])
        result = var.f_bar({"Z": z})

        expected = 1.0 + 2.0 * z
        assert torch.allclose(result, expected)

    def test_f_bar_multiple_parents(self) -> None:
        """Test f_bar with multiple parents computes correct linear combination."""
        var = LinearVariable(
            name="Y",
            parent_names=["X", "Z"],
            sigma=1.0,
            coefs={"X": 2.0, "Z": 3.0},
            intercept=0.5,
        )

        x = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5, 1.0])
        result = var.f_bar({"X": x, "Z": z})

        expected = 0.5 + 2.0 * x + 3.0 * z
        assert torch.allclose(result, expected)

    def test_f_computes_value_with_noise(self) -> None:
        """Test f adds noise correctly to f_bar."""
        var = LinearVariable(
            name="X",
            parent_names=["Z"],
            sigma=2.0,
            coefs={"Z": 1.0},
            intercept=0.0,
        )

        z = torch.tensor([1.0, 2.0, 3.0])
        u = torch.tensor([0.5, -0.5, 1.0])
        result = var.f({"Z": z}, u)

        expected = z + 2.0 * u
        assert torch.allclose(result, expected)

    def test_f_with_precomputed_f_bar(self) -> None:
        """Test f uses precomputed f_bar when provided."""
        var = LinearVariable(
            name="X",
            parent_names=["Z"],
            sigma=2.0,
            coefs={"Z": 1.0},
            intercept=0.0,
        )

        precomputed_f_bar = torch.tensor([10.0, 20.0, 30.0])
        u = torch.tensor([1.0, 1.0, 1.0])
        result = var.f({}, u, f_bar=precomputed_f_bar)

        expected = precomputed_f_bar + 2.0 * u
        assert torch.allclose(result, expected)

    def test_f_without_precomputed_f_bar(self) -> None:
        """Test f computes f_bar when not provided."""
        var = LinearVariable(
            name="X",
            parent_names=["Z"],
            sigma=1.0,
            coefs={"Z": 2.0},
            intercept=1.0,
        )

        z = torch.tensor([1.0, 2.0])
        u = torch.tensor([0.0, 0.0])
        result = var.f({"Z": z}, u)

        expected = 1.0 + 2.0 * z  # f_bar + sigma * u (u=0)
        assert torch.allclose(result, expected)

    def test_sigma_zero_produces_deterministic_output(self) -> None:
        """Test that sigma=0 produces deterministic (no noise) output."""
        var = LinearVariable(
            name="X",
            parent_names=[],
            sigma=0.0,
            coefs={},
            intercept=5.0,
        )

        u = torch.tensor([100.0, -100.0, 0.0])  # Noise should be ignored
        result = var.f({}, u)

        expected = torch.tensor([5.0, 5.0, 5.0])
        assert torch.allclose(result, expected)

    def test_parent_names_inferred_from_coefs(self) -> None:
        """Test that parent_names is inferred from coefs keys when not provided."""
        var = LinearVariable(
            name="X",
            sigma=1.0,
            coefs={"A": 1.0, "B": 2.0},
        )

        assert set(var.parent_names) == {"A", "B"}

    def test_parent_names_defaults_to_empty_when_no_coefs(self) -> None:
        """Test that parent_names defaults to empty list when coefs not provided."""
        var = LinearVariable(
            name="X",
            sigma=1.0,
        )

        assert var.parent_names == []
        assert var._coefs == {}

    def test_raises_when_coefs_missing_parent(self) -> None:
        """Test that ValueError is raised when coefs is missing a parent."""
        with pytest.raises(ValueError, match="missing coefficients for"):
            LinearVariable(
                name="X",
                sigma=1.0,
                parent_names=["A", "B"],
                coefs={"A": 1.0},  # Missing "B"
            )

    def test_raises_when_coefs_has_extra_key(self) -> None:
        """Test that ValueError is raised when coefs has extra keys."""
        with pytest.raises(ValueError, match="extra coefficients for"):
            LinearVariable(
                name="X",
                sigma=1.0,
                parent_names=["A"],
                coefs={"A": 1.0, "B": 2.0},  # Extra "B"
            )

    def test_raises_when_coefs_and_parent_names_mismatch(self) -> None:
        """Test that ValueError is raised when coefs keys don't match parent_names."""
        with pytest.raises(
            ValueError, match="Coefficient keys must match parent_names"
        ):
            LinearVariable(
                name="X",
                sigma=1.0,
                parent_names=["A", "B"],
                coefs={"A": 1.0, "C": 2.0},  # "C" instead of "B"
            )
