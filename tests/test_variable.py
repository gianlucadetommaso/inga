"""Tests for Variable and LinearVariable classes."""

import pytest
import torch
from inga.variable.base import Variable
from inga.variable.linear import LinearVariable
from inga.variable.functional import FunctionalVariable


class ConcreteVariable(Variable):
    """Concrete implementation of Variable for testing."""

    def f_mean(self, parents: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def test_base_variable_f_mean_not_implemented(self) -> None:
        """Bare Variable should not implement a mean function."""
        var = Variable(name="X")

        with pytest.raises(NotImplementedError, match="structural function"):
            var.f_mean({})

    def test_base_variable_f_not_implemented(self) -> None:
        """Bare Variable should not implement a full structural equation."""
        var = Variable(name="X", sigma=1.0)

        with pytest.raises(NotImplementedError, match="structural equation"):
            var.f({}, torch.randn(3))


class TestLinearVariable:
    """Tests for LinearVariable class."""

    def test_init_stores_attributes(self) -> None:
        """Test that __init__ correctly stores all attributes."""
        name = "X"
        parent_names = ["Z"]
        sigma = 2.0
        coefs = {"Z": 1.5}
        intercept = 0.5

        var = LinearVariable(
            name=name,
            parent_names=parent_names,
            sigma=sigma,
            coefs=coefs,
            intercept=intercept,
        )

        assert var.name == name
        assert list(var.parent_names) == parent_names
        assert var.sigma == sigma
        assert var._coefs == coefs
        assert var._intercept == intercept

    def test_init_no_parents(self) -> None:
        """Test initialization with no parents (root node)."""
        var = LinearVariable(
            name="Z",
            sigma=1.0,
        )

        assert var.name == "Z"
        assert list(var.parent_names) == []
        assert var._coefs == {}

    @pytest.mark.parametrize("intercept", [0.0, 3.0, -1.5])
    def test_f_mean_no_parents(self, intercept: float) -> None:
        """Test f_mean with no parents returns intercept."""
        var = LinearVariable(
            name="Z",
            sigma=1.0,
            intercept=intercept,
        )

        result = var.f_mean({})
        assert result == intercept

    @pytest.mark.parametrize(
        "coef_z,intercept",
        [
            (2.0, 1.0),
            (0.5, 0.0),
            (-1.0, 2.5),
        ],
    )
    def test_f_mean_one_parent(self, coef_z: float, intercept: float) -> None:
        """Test f_mean with one parent computes correct linear combination."""
        var = LinearVariable(
            name="X",
            sigma=1.0,
            coefs={"Z": coef_z},
            intercept=intercept,
        )

        z = torch.tensor([1.0, 2.0, 3.0])
        result = var.f_mean({"Z": z})

        expected = intercept + coef_z * z
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        "coef_x,coef_z,intercept",
        [
            (2.0, 3.0, 0.5),
            (1.0, 1.0, 0.0),
            (-0.5, 2.0, 1.0),
        ],
    )
    def test_f_mean_multiple_parents(
        self, coef_x: float, coef_z: float, intercept: float
    ) -> None:
        """Test f_mean with multiple parents computes correct linear combination."""
        var = LinearVariable(
            name="Y",
            sigma=1.0,
            coefs={"X": coef_x, "Z": coef_z},
            intercept=intercept,
        )

        x = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5, 1.0])
        result = var.f_mean({"X": x, "Z": z})

        expected = intercept + coef_x * x + coef_z * z
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        "sigma,coef_z,intercept",
        [
            (2.0, 1.0, 0.0),
            (0.5, 2.0, 1.0),
            (1.0, -1.0, 0.5),
        ],
    )
    def test_f_computes_value_with_noise(
        self, sigma: float, coef_z: float, intercept: float
    ) -> None:
        """Test f adds noise correctly to f_mean."""
        var = LinearVariable(
            name="X",
            sigma=sigma,
            coefs={"Z": coef_z},
            intercept=intercept,
        )

        z = torch.tensor([1.0, 2.0, 3.0])
        u = torch.tensor([0.5, -0.5, 1.0])
        result = var.f({"Z": z}, u)

        f_mean = intercept + coef_z * z
        expected = f_mean + sigma * u
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("sigma", [1.0, 2.0, 0.5])
    def test_f_with_precomputed_f_mean(self, sigma: float) -> None:
        """Test f uses precomputed f_mean when provided."""
        var = LinearVariable(
            name="X",
            sigma=sigma,
            coefs={"Z": 1.0},
        )

        precomputed_f_mean = torch.tensor([10.0, 20.0, 30.0])
        u = torch.tensor([1.0, 1.0, 1.0])
        result = var.f({}, u, f_mean=precomputed_f_mean)

        expected = precomputed_f_mean + sigma * u
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        "sigma,coef_z,intercept",
        [
            (1.0, 2.0, 1.0),
            (0.0, 1.0, 0.0),
            (2.0, -1.0, 0.5),
        ],
    )
    def test_f_without_precomputed_f_mean(
        self, sigma: float, coef_z: float, intercept: float
    ) -> None:
        """Test f computes f_mean when not provided."""
        var = LinearVariable(
            name="X",
            sigma=sigma,
            coefs={"Z": coef_z},
            intercept=intercept,
        )

        z = torch.tensor([1.0, 2.0])
        u = torch.tensor([0.5, -0.5])
        result = var.f({"Z": z}, u)

        f_mean = intercept + coef_z * z
        expected = f_mean + sigma * u
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("intercept", [5.0, 0.0, -3.0])
    def test_sigma_zero_produces_deterministic_output(self, intercept: float) -> None:
        """Test that sigma=0 produces deterministic (no noise) output."""
        var = LinearVariable(
            name="X",
            sigma=0.0,
            intercept=intercept,
        )

        u = torch.tensor([100.0, -100.0, 0.0])  # Noise should be ignored
        result = var.f({}, u)

        expected = torch.full_like(u, intercept)
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        "coefs",
        [
            {"A": 1.0, "B": 2.0},
            {"X": 0.5},
            {"P1": 1.0, "P2": 2.0, "P3": 3.0},
        ],
    )
    def test_parent_names_inferred_from_coefs(self, coefs: dict[str, float]) -> None:
        """Test that parent_names is inferred from coefs keys when not provided."""
        var = LinearVariable(
            name="X",
            sigma=1.0,
            coefs=coefs,
        )

        assert set(var.parent_names) == set(coefs.keys())

    def test_parent_names_defaults_to_empty_when_no_coefs(self) -> None:
        """Test that parent_names defaults to empty list when coefs not provided."""
        var = LinearVariable(
            name="X",
            sigma=1.0,
        )

        assert var.parent_names == []
        assert var._coefs == {}

    @pytest.mark.parametrize(
        "parent_names,coefs,error_match",
        [
            (["A", "B"], {"A": 1.0}, "missing coefficients for"),
            (["A"], {"A": 1.0, "B": 2.0}, "extra coefficients for"),
            (
                ["A", "B"],
                {"A": 1.0, "C": 2.0},
                "Coefficient keys must match parent_names",
            ),
        ],
    )
    def test_raises_when_coefs_parent_names_mismatch(
        self, parent_names: list[str], coefs: dict[str, float], error_match: str
    ) -> None:
        """Test that ValueError is raised when coefs keys don't match parent_names."""
        with pytest.raises(ValueError, match=error_match):
            LinearVariable(
                name="X",
                sigma=1.0,
                parent_names=parent_names,
                coefs=coefs,
            )


class TestFunctionalVariable:
    """Tests for FunctionalVariable class."""

    def test_f_mean_no_parents(self) -> None:
        """Test f_mean returns mean function output with no parents."""
        var = FunctionalVariable(
            name="X",
            sigma=1.0,
            f_mean=lambda _: torch.tensor(2.0),
        )

        result = var.f_mean({})
        assert result == 2.0

    def test_f_mean_with_parents(self) -> None:
        """Test f_mean computes a nonlinear transform of parents."""
        var = FunctionalVariable(
            name="Y",
            sigma=1.0,
            parent_names=["X"],
            f_mean=lambda parents: torch.sin(parents["X"]),
        )

        x = torch.tensor([0.0, 1.0])
        result = var.f_mean({"X": x})
        assert torch.allclose(result, torch.sin(x))
