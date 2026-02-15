"""Tests for the SEM class."""

import matplotlib
import torch
import pytest
from steindag.variable.base import Variable
from steindag.variable.linear import LinearVariable
from steindag.sem.base import SEM
from steindag.sem.random import (
    RandomSEMConfig,
    _build_f_mean,
    random_sem,
    resolve_transforms,
)

matplotlib.use("Agg")


@pytest.fixture
def simple_sem() -> SEM:
    """Create a simple SEM with Z -> X structure."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
            ),
        ]
    )


@pytest.fixture
def chain_sem() -> SEM:
    """Create a chain SEM: Z -> X -> Y."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=["Z"], sigma=1.0, coefs={"Z": 1.0}, intercept=0.0
            ),
            LinearVariable(
                name="Y", parent_names=["X"], sigma=1.0, coefs={"X": 2.0}, intercept=0.0
            ),
        ]
    )


@pytest.fixture
def collider_sem() -> SEM:
    """Create a collider SEM: Z -> Y <- X."""
    return SEM(
        variables=[
            LinearVariable(
                name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="X", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
            ),
            LinearVariable(
                name="Y",
                parent_names=["Z", "X"],
                sigma=1.0,
                coefs={"Z": 1.0, "X": 2.0},
                intercept=0.0,
            ),
        ]
    )


class TestSEMInit:
    """Tests for SEM initialization."""

    def test_init_stores_variables(self, simple_sem: SEM) -> None:
        """Test that __init__ correctly stores variables by name."""
        assert "Z" in simple_sem._variables
        assert "X" in simple_sem._variables
        assert len(simple_sem._variables) == 2

    def test_init_creates_posterior(self, simple_sem: SEM) -> None:
        """Test that __init__ creates a LaplacePosterior."""
        assert simple_sem.posterior is not None

    def test_init_passes_posterior_kwargs(self) -> None:
        """Test that SEM forwards posterior configuration to LaplacePosterior."""
        sem = SEM(
            variables=[
                LinearVariable(
                    name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
                LinearVariable(
                    name="X",
                    parent_names=["Z"],
                    sigma=1.0,
                    coefs={"Z": 1.0},
                    intercept=0.0,
                ),
            ],
            posterior_kwargs={
                "num_map_restarts": 4,
                "num_mixture_components": 2,
                "adam_lr": 1e-2,
            },
        )

        assert sem.posterior._num_map_restarts == 4
        assert sem.posterior._num_mixture_components == 2
        assert sem.posterior._adam_lr == 1e-2

    def test_init_accepts_parent_only_variables_without_sigma(self) -> None:
        """SEM should accept DAG-only variable declarations."""
        sem = SEM(
            variables=[
                Variable(name="Z"),
                Variable(name="X", parent_names=["Z"]),
            ]
        )

        assert set(sem._variables.keys()) == {"Z", "X"}


class TestSEMGenerate:
    """Tests for SEM.generate method."""

    def test_generate_returns_all_variables(self, simple_sem: SEM) -> None:
        """Test that generate returns values for all variables."""
        torch.manual_seed(42)
        values = simple_sem.generate(10)

        assert "Z" in values
        assert "X" in values
        assert len(values) == 2

    def test_generate_correct_shape(self, simple_sem: SEM) -> None:
        """Test that generated values have correct shape."""
        torch.manual_seed(42)
        values = simple_sem.generate(100)

        assert values["Z"].shape == (100,)
        assert values["X"].shape == (100,)

    def test_generate_respects_causal_structure(self, chain_sem: SEM) -> None:
        """Test that generate respects the causal structure."""
        torch.manual_seed(42)
        values = chain_sem.generate(1000)

        # Y should be correlated with X (Y = 2*X + noise)
        # Correlation should be positive and significant
        corr = torch.corrcoef(torch.stack([values["X"], values["Y"]]))[0, 1]
        assert corr > 0.5

    def test_generate_reproducible_with_seed(self, simple_sem: SEM) -> None:
        """Test that generate is reproducible with the same seed."""
        torch.manual_seed(42)
        values1 = simple_sem.generate(10)

        torch.manual_seed(42)
        values2 = simple_sem.generate(10)

        assert torch.allclose(values1["Z"], values2["Z"])
        assert torch.allclose(values1["X"], values2["X"])

    def test_generate_collider_structure(self, collider_sem: SEM) -> None:
        """Test generate with collider structure (independent causes)."""
        torch.manual_seed(42)
        values = collider_sem.generate(1000)

        # Z and X should be approximately uncorrelated (independent causes)
        corr = torch.corrcoef(torch.stack([values["Z"], values["X"]]))[0, 1]
        assert abs(corr) < 0.1

    def test_generate_single_sample(self, simple_sem: SEM) -> None:
        """Test generate with a single sample."""
        torch.manual_seed(42)
        values = simple_sem.generate(1)

        assert values["Z"].shape == (1,)
        assert values["X"].shape == (1,)


class TestSEMPosterior:
    """Tests for SEM posterior inference."""

    def test_posterior_fit_and_sample(self, simple_sem: SEM) -> None:
        """Test that posterior.fit and posterior.sample work."""
        torch.manual_seed(42)
        values = simple_sem.generate(10)

        observed = {"X": values["X"]}
        simple_sem.posterior.fit(observed)

        torch.manual_seed(0)
        samples = simple_sem.posterior.sample(100)

        assert "Z" in samples
        assert samples["Z"].shape == (10, 100)

    def test_posterior_sample_requires_fit(self, simple_sem: SEM) -> None:
        """Test that sample raises error if fit not called."""
        # Create a fresh SEM to ensure no state
        sem = SEM(
            variables=[
                LinearVariable(
                    name="Z", parent_names=[], sigma=1.0, coefs={}, intercept=0.0
                ),
            ]
        )

        with pytest.raises(ValueError, match="fit"):
            sem.posterior.sample(10)


class TestSEMDraw:
    """Tests for SEM.draw visualization."""

    def test_draw_creates_png_file(self, tmp_path) -> None:
        """Draw API should render and save a PNG file."""
        sem = SEM(
            variables=[
                Variable(name="V1"),
                Variable(name="X", parent_names=["V1"]),
                Variable(name="Y", parent_names=["X"]),
            ]
        )

        out = sem.draw(
            output_path=tmp_path / "dag.png",
            observed_names=["X"],
            title="DAG",
        )

        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 0

    def test_draw_rejects_unknown_observed_variable(self) -> None:
        """Unknown observed names should raise ValueError in draw."""
        sem = SEM(
            variables=[
                Variable(name="V1"),
                Variable(name="X", parent_names=["V1"]),
            ]
        )

        with pytest.raises(ValueError, match="Unknown observed variable"):
            sem.draw(output_path="plots/test_draw.png", observed_names=["NOPE"])


class TestSEMPlotDAG:
    """Tests for SEM.plot_dag visualization."""

    def test_plot_dag_labels_and_shading(self, chain_sem: SEM) -> None:
        """Node labels should be present and observed nodes should be shaded."""
        _, ax = chain_sem.plot_dag(observed_names=["X"])

        labels = {text.get_text() for text in ax.texts}
        assert labels == {"Z", "X", "Y"}

        node_patches = [
            patch
            for patch in ax.patches
            if getattr(patch, "get_gid", lambda: None)() is not None
            and str(patch.get_gid()).startswith("node:")
        ]
        assert len(node_patches) == 3

        patch_by_name = {
            str(patch.get_gid()).split(":", 1)[1]: patch for patch in node_patches
        }

        observed_facecolor = patch_by_name["X"].get_facecolor()[:3]
        unobserved_facecolor = patch_by_name["Z"].get_facecolor()[:3]

        assert observed_facecolor == pytest.approx((0.741, 0.741, 0.741), abs=0.02)
        assert unobserved_facecolor == pytest.approx((1.0, 1.0, 1.0), abs=0.01)

    def test_plot_dag_rejects_unknown_observed_variable(self, simple_sem: SEM) -> None:
        """Unknown observed names should raise a ValueError."""
        with pytest.raises(ValueError, match="Unknown observed variable"):
            simple_sem.plot_dag(observed_names=["NOT_IN_SEM"])

    def test_plot_dag_highlights_path_categories(self) -> None:
        """Path categories should be color-coded for treatment/outcome queries."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "U", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "T", sigma=1.0, parent_names=["U"], coefs={"U": 1.0}, intercept=0.0
                ),
                LinearVariable(
                    "M", sigma=1.0, parent_names=["T"], coefs={"T": 1.0}, intercept=0.0
                ),
                LinearVariable(
                    "B", sigma=1.0, parent_names=["T"], coefs={"T": 1.0}, intercept=0.0
                ),
                LinearVariable(
                    "Y",
                    sigma=1.0,
                    parent_names=["M", "B", "U"],
                    coefs={"M": 1.0, "B": 1.0, "U": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        _, ax = sem.plot_dag(
            observed_names=["T", "B"],
            treatment_name="T",
            outcome_name="Y",
        )

        edge_patches = [
            patch
            for patch in ax.patches
            if getattr(patch, "get_gid", lambda: None)() is not None
            and str(patch.get_gid()).startswith("edge:")
        ]
        edge_categories_by_name: dict[str, set[str]] = {}
        for patch in edge_patches:
            gid = str(patch.get_gid())
            _, edge_name, category = gid.split(":", 2)
            edge_categories_by_name.setdefault(edge_name, set()).add(category)

        assert "open_causal" in edge_categories_by_name["T->M"]
        assert "open_causal" in edge_categories_by_name["M->Y"]
        assert "open_noncausal" in edge_categories_by_name["U->T"]
        assert "open_noncausal" in edge_categories_by_name["U->Y"]
        assert "blocked_causal" in edge_categories_by_name["T->B"]
        assert "blocked_causal" in edge_categories_by_name["B->Y"]

        node_patches = [
            patch
            for patch in ax.patches
            if getattr(patch, "get_gid", lambda: None)() is not None
            and str(patch.get_gid()).startswith("node:")
        ]
        node_facecolor_by_name = {
            str(p.get_gid()).split(":", 1)[1]: p.get_facecolor()[:3]
            for p in node_patches
        }
        assert node_facecolor_by_name["T"] == pytest.approx((0.18, 0.36, 1.0), abs=0.08)
        assert node_facecolor_by_name["Y"] == pytest.approx(
            (0.56, 0.27, 0.68), abs=0.08
        )

    def test_plot_dag_supports_multiple_edge_categories(self) -> None:
        """An edge can carry multiple path categories and should be duplicated."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "U", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "T", sigma=1.0, parent_names=["U"], coefs={"U": 1.0}, intercept=0.0
                ),
                LinearVariable(
                    "M",
                    sigma=1.0,
                    parent_names=["T", "U"],
                    coefs={"T": 1.0, "U": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "Y",
                    sigma=1.0,
                    parent_names=["M", "U"],
                    coefs={"M": 1.0, "U": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        _, ax = sem.plot_dag(
            observed_names=["T", "M"],
            treatment_name="T",
            outcome_name="Y",
        )

        edge_patches = [
            patch
            for patch in ax.patches
            if getattr(patch, "get_gid", lambda: None)() is not None
            and str(patch.get_gid()).startswith("edge:")
        ]
        categories_t_to_m = {
            str(p.get_gid()).split(":", 2)[2]
            for p in edge_patches
            if str(p.get_gid()).split(":", 2)[1] == "T->M"
        }
        assert categories_t_to_m == {"blocked_causal", "open_noncausal"}

    def test_plot_dag_requires_treatment_observed(self, simple_sem: SEM) -> None:
        """Treatment must be included in observed_names when provided."""
        with pytest.raises(ValueError, match="treatment_name"):
            simple_sem.plot_dag(
                observed_names=[],
                treatment_name="X",
                outcome_name="Z",
            )

    def test_plot_dag_requires_outcome_unobserved(self, simple_sem: SEM) -> None:
        """Outcome cannot be included in observed_names when provided."""
        with pytest.raises(ValueError, match="outcome_name"):
            simple_sem.plot_dag(
                observed_names=["X", "Z"],
                treatment_name="X",
                outcome_name="Z",
            )

    def test_path_flows_do_not_add_non_simple_collider_descendant_walks(self) -> None:
        """Collider-descendant motifs should not create non-simple repeated-node paths."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "T", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "Y",
                    sigma=1.0,
                    parent_names=["T", "P"],
                    coefs={"T": 1.0, "P": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "P", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "D",
                    sigma=1.0,
                    parent_names=["Y", "P"],
                    coefs={"Y": 1.0, "P": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        flows = sem.path_flows(
            treatment_name="T",
            outcome_name="Y",
            observed_names=["T", "D"],
        )

        assert ["T", "Y", "D", "P", "Y"] not in flows["open_noncausal"]

    def test_path_flows_exclude_repeated_node_detours(self) -> None:
        """Synthetic detours must remain simple paths (no repeated nodes)."""
        from steindag.sem.random import RandomSEMConfig, random_sem

        sem = random_sem(
            RandomSEMConfig(
                num_variables=6,
                parent_prob=0.6,
                nonlinear_prob=0.6,
                seed=11,
            )
        )
        flows = sem.path_flows(
            treatment_name="X1",
            outcome_name="X4",
            observed_names=["X1", "X3", "X5"],
        )

        assert ["X1", "X4", "X5", "X3", "X4"] not in flows["open_noncausal"]
        assert ["X1", "X2", "X4", "X5", "X3", "X4"] not in flows["open_noncausal"]

    def test_animate_flow_gif_creates_file(self, tmp_path) -> None:
        """Core GIF animation API should render successfully."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "X0", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "X1",
                    sigma=1.0,
                    parent_names=["X0"],
                    coefs={"X0": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "X2",
                    sigma=1.0,
                    parent_names=["X1"],
                    coefs={"X1": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        out = sem.animate_flow_gif(
            output_path=tmp_path / "flow.gif",
            observed_names=["X1"],
            treatment_name="X1",
            outcome_name="X2",
            fps=8,
            frames_per_flow=8,
            title="Flow",
        )

        assert out.exists()
        assert out.stat().st_size > 0

    def test_flow_animation_stages_include_directed_causal_first(self) -> None:
        """Staged animation should begin with directed causal flow stage."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "X0", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "X1",
                    sigma=1.0,
                    parent_names=["X0"],
                    coefs={"X0": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "X2",
                    sigma=1.0,
                    parent_names=["X1"],
                    coefs={"X1": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        stages = sem.flow_animation_stages(
            treatment_name="X1",
            outcome_name="X2",
            observed_names=["X1"],
        )
        assert len(stages) >= 1
        assert stages[0]["name"] == "directed_causal"
        assert stages[0]["color"] == "#2ECC71"

    def test_flow_animation_stages_include_confounder_bias(self) -> None:
        """Latent confounders should produce red confounder-bias stages."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "U", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "T", sigma=1.0, parent_names=["U"], coefs={"U": 1.0}, intercept=0.0
                ),
                LinearVariable(
                    "Y",
                    sigma=1.0,
                    parent_names=["T", "U"],
                    coefs={"T": 1.0, "U": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        stages = sem.flow_animation_stages(
            treatment_name="T",
            outcome_name="Y",
            observed_names=["T"],
        )

        conf_stages = [
            stage for stage in stages if str(stage["name"]).startswith("confounder:")
        ]
        assert len(conf_stages) == 1
        assert conf_stages[0]["color"] == "#FF4D4F"
        assert conf_stages[0]["pulse_nodes"] == ["U"]

    def test_flow_animation_stages_classifies_observed_mediator_not_selection(
        self,
    ) -> None:
        """Observed mediators should be labeled as mediator bias, not selection bias."""
        sem = SEM(
            variables=[
                LinearVariable(
                    "V1", sigma=1.0, parent_names=[], coefs={}, intercept=0.0
                ),
                LinearVariable(
                    "X",
                    sigma=1.0,
                    parent_names=["V1"],
                    coefs={"V1": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "V2",
                    sigma=1.0,
                    parent_names=["V1", "X"],
                    coefs={"V1": 1.0, "X": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "V3",
                    sigma=1.0,
                    parent_names=["V1", "X", "V2"],
                    coefs={"V1": 1.0, "X": 1.0, "V2": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "Y",
                    sigma=1.0,
                    parent_names=["V1", "X", "V2", "V3"],
                    coefs={"V1": 1.0, "X": 1.0, "V2": 1.0, "V3": 1.0},
                    intercept=0.0,
                ),
                LinearVariable(
                    "V4",
                    sigma=1.0,
                    parent_names=["V1", "X", "V2", "V3", "Y"],
                    coefs={"V1": 1.0, "X": 1.0, "V2": 1.0, "V3": 1.0, "Y": 1.0},
                    intercept=0.0,
                ),
            ]
        )

        stages = sem.flow_animation_stages(
            treatment_name="X",
            outcome_name="Y",
            observed_names=["X", "V2", "V4"],
        )

        directed_stage = next(
            stage for stage in stages if stage["name"] == "directed_causal"
        )
        directed_paths = {tuple(path) for path in directed_stage["paths"]}
        assert ("X", "Y") in directed_paths
        assert ("X", "V2", "V3", "Y") in directed_paths
        assert len(directed_paths) > 1

        mediator_stages = [
            stage for stage in stages if str(stage["name"]).startswith("mediator:")
        ]
        assert len(mediator_stages) == 1
        assert mediator_stages[0]["name"] == "mediator:V2"
        assert mediator_stages[0]["pulse_nodes"] == ["V2"]

        selection_stage = next(
            stage for stage in stages if stage["name"] == "selection_bias"
        )
        assert selection_stage["pulse_nodes"] == ["V4"]


class TestRandomSEM:
    """Tests for random SEM generation."""

    def test_random_sem_generates_dag(self) -> None:
        """Ensure random SEM respects DAG ordering."""
        config = RandomSEMConfig(num_variables=6, parent_prob=0.6, seed=123)
        sem = random_sem(config)
        assert len(sem._variables) == 6

        for idx, (name, variable) in enumerate(sem._variables.items()):
            allowed_parents = {f"X{i}" for i in range(idx)}
            assert set(variable.parent_names).issubset(allowed_parents)

    def test_random_sem_reproducible(self) -> None:
        """Ensure the random SEM is reproducible with a seed."""
        config = RandomSEMConfig(num_variables=4, parent_prob=0.5, seed=99)
        sem1 = random_sem(config)
        sem2 = random_sem(config)

        for (name1, var1), (name2, var2) in zip(
            sem1._variables.items(), sem2._variables.items()
        ):
            assert name1 == name2
            assert var1.parent_names == var2.parent_names

    def test_random_sem_rejects_invalid_prob(self) -> None:
        """Ensure invalid parent probabilities raise errors."""
        with pytest.raises(ValueError, match="parent_prob"):
            random_sem(RandomSEMConfig(num_variables=3, parent_prob=1.5))

    def test_random_sem_rejects_non_positive_size(self) -> None:
        """Ensure invalid num_variables raises errors."""
        with pytest.raises(ValueError, match="num_variables"):
            random_sem(RandomSEMConfig(num_variables=0))

    def test_random_sem_includes_nonlinear_variables(self) -> None:
        """Ensure nonlinear option creates at least one FunctionalVariable when forced."""
        from steindag.variable.functional import FunctionalVariable

        config = RandomSEMConfig(
            num_variables=4,
            parent_prob=0.7,
            nonlinear_prob=1.0,
            seed=7,
        )
        sem = random_sem(config)
        assert any(
            isinstance(var, FunctionalVariable) for var in sem._variables.values()
        )

    def test_build_f_mean_normalizes_linear_predictor_gradient(self) -> None:
        """Linear pre-activation gradient should be coefficient-norm standardized."""
        f_mean = _build_f_mean(
            parent_names=["A", "B"],
            coefs={"A": 3.0, "B": 4.0},
            intercept=0.0,
            transforms=None,
        )

        a = torch.tensor(0.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        out = f_mean({"A": a, "B": b})

        da, db = torch.autograd.grad(out, [a, b])
        grad_norm = torch.sqrt(da**2 + db**2)

        expected_norm = torch.tensor((25.0 / 26.0) ** 0.5)
        assert torch.allclose(grad_norm, expected_norm, atol=1e-6)
        assert torch.allclose(da, torch.tensor(3.0 / (26.0**0.5)), atol=1e-6)
        assert torch.allclose(db, torch.tensor(4.0 / (26.0**0.5)), atol=1e-6)

    def test_random_sem_seed22_generate_is_finite_regression(self) -> None:
        """Regression: seed-22 random SEM generation should not emit NaN/Inf."""
        sem = random_sem(
            RandomSEMConfig(
                num_variables=6,
                parent_prob=0.6,
                nonlinear_prob=0.8,
                sigma_range=(0.7, 1.2),
                coef_range=(-1.0, 1.0),
                intercept_range=(-0.5, 0.5),
                seed=22,
            )
        )
        data = sem.generate(768)
        for values in data.values():
            assert torch.isfinite(values).all()

    def test_resolve_transforms_supports_smooth_and_sharp_families(self) -> None:
        """Ensure broader transform pool includes smooth and sharp relations."""
        smooth_names = ["sigmoid", "softsign", "atan", "swish", "gelu"]
        sharp_names = ["relu", "leaky_relu", "elu", "softplus_sharp", "abs"]

        x = torch.linspace(-3.0, 3.0, steps=17)
        for fn in resolve_transforms([*smooth_names, *sharp_names, "cubic"]):
            y = fn(x)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()
