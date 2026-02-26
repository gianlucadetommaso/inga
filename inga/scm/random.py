"""Random SCM generator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import random
import torch
from torch import Tensor
from torch.nn import functional as F

from inga.scm.base import SCM
from inga.scm.variable.categorical import CategoricalVariable
from inga.scm.variable.base import Variable
from inga.scm.variable.linear import LinearVariable
from inga.scm.variable.functional import FunctionalVariable


@dataclass(frozen=True)
class RandomSCMConfig:
    """Configuration for random SCM generation."""

    num_variables: int
    parent_prob: float = 0.3
    sigma_range: tuple[float, float] = (0.5, 1.5)
    coef_range: tuple[float, float] = (-2.0, 2.0)
    intercept_range: tuple[float, float] = (-1.0, 1.0)
    nonlinear_prob: float = 0.5
    categorical_prob: float = 0.0
    num_categories_range: tuple[int, int] = (2, 6)
    categorical_temperature_range: tuple[float, float] = (0.05, 0.8)
    categorical_weight_range: tuple[float, float] = (-2.0, 2.0)
    categorical_bias_range: tuple[float, float] = (-1.0, 1.0)
    seed: int | None = None


class RandomCategoricalVariable(CategoricalVariable):
    """Parametric categorical variable used by random SCM generation.

    Logits are generated from a flexible mixed-parent model supporting both
    scalar (Gaussian) and vector (categorical one-hot/probability) parents.
    """

    def __init__(
        self,
        name: str,
        parent_names: list[str],
        parent_kinds: dict[str, Literal["gaussian", "categorical"]],
        num_categories: int,
        temperature: float,
        bias: list[float],
        gaussian_parent_weights: dict[str, list[float]],
        categorical_parent_weights: dict[str, list[list[float]]],
        transforms: list[str] | None = None,
    ) -> None:
        super().__init__(name=name, parent_names=parent_names, temperature=temperature)
        self._num_categories = num_categories
        self._parent_kinds = dict(parent_kinds)
        self._bias = bias
        self._gaussian_parent_weights = gaussian_parent_weights
        self._categorical_parent_weights = categorical_parent_weights
        self._transforms = transforms or []

    def f_logits(self, parents: dict[str, Tensor]) -> Tensor:
        parent_reference = next(iter(parents.values()), None)
        if parent_reference is None:
            logits = torch.tensor(self._bias)
        else:
            logits = torch.tensor(
                self._bias,
                dtype=parent_reference.dtype,
                device=parent_reference.device,
            ).expand(len(parent_reference), self._num_categories)

        for parent_name in self.parent_names:
            parent_value = parents[parent_name]
            parent_kind = self._parent_kinds[parent_name]

            if parent_kind == "gaussian":
                weights = torch.tensor(
                    self._gaussian_parent_weights[parent_name],
                    dtype=parent_value.dtype,
                    device=parent_value.device,
                )
                logits = logits + parent_value.unsqueeze(-1) * weights
                continue

            weight_matrix = torch.tensor(
                self._categorical_parent_weights[parent_name],
                dtype=parent_value.dtype,
                device=parent_value.device,
            )
            logits = logits + parent_value @ weight_matrix

        if self._transforms:
            transforms = resolve_transforms(self._transforms)
            for transform in transforms:
                logits = transform(logits)
        return logits

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parent_names": list(self.parent_names),
            "parent_kinds": dict(self._parent_kinds),
            "num_categories": self._num_categories,
            "temperature": self._temperature,
            "bias": list(self._bias),
            "gaussian_parent_weights": {
                key: list(value)
                for key, value in self._gaussian_parent_weights.items()
            },
            "categorical_parent_weights": {
                key: [list(row) for row in value]
                for key, value in self._categorical_parent_weights.items()
            },
            "transforms": list(self._transforms),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RandomCategoricalVariable":
        return cls(
            name=str(payload["name"]),
            parent_names=[str(name) for name in payload.get("parent_names", [])],
            parent_kinds={
                str(key): str(value)  # type: ignore[dict-item]
                for key, value in dict(payload.get("parent_kinds", {})).items()
            },
            num_categories=int(payload["num_categories"]),
            temperature=float(payload["temperature"]),
            bias=[float(value) for value in list(payload["bias"])],
            gaussian_parent_weights={
                str(key): [float(v) for v in list(values)]
                for key, values in dict(payload.get("gaussian_parent_weights", {})).items()
            },
            categorical_parent_weights={
                str(key): [
                    [float(v) for v in list(row)]
                    for row in list(values)
                ]
                for key, values in dict(payload.get("categorical_parent_weights", {})).items()
            },
            transforms=[str(name) for name in list(payload.get("transforms", []))],
        )


def _compose_transforms(
    base: Tensor,
    transforms: Iterable[Callable[[Tensor], Tensor]],
) -> Tensor:
    """Compose a sequence of transformations onto a base tensor."""
    value = base
    for transform in transforms:
        value = transform(value)
    return value


_TRANSFORM_MAP: dict[str, Callable[[Tensor], Tensor]] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softsign": F.softsign,
    "atan": torch.atan,
    "swish": lambda x: x * torch.sigmoid(x),
    "gelu": F.gelu,
    "relu": torch.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.1),
    "elu": F.elu,
    "softplus_sharp": lambda x: F.softplus(x, beta=8.0),
    "abs": torch.abs,
    "cubic": lambda x: x + 0.25 * x**3,
}


def _sample_transforms(rng: random.Random) -> list[Callable[[Tensor], Tensor]]:
    """Sample a list of nonlinear transforms."""
    transform_pool = list(_TRANSFORM_MAP.values())
    num_transforms = rng.randint(1, 3)
    return rng.sample(transform_pool, k=num_transforms)


def _sample_transform_names(rng: random.Random) -> list[str]:
    """Sample a list of nonlinear transform names."""
    transform_pool = list(_TRANSFORM_MAP.keys())
    num_transforms = rng.randint(1, 3)
    return rng.sample(transform_pool, k=num_transforms)


def resolve_transforms(names: Iterable[str]) -> list[Callable[[Tensor], Tensor]]:
    """Resolve transform names to callables."""
    transforms: list[Callable[[Tensor], Tensor]] = []
    for name in names:
        if name not in _TRANSFORM_MAP:
            raise ValueError(f"Unknown transform '{name}'.")
        transforms.append(_TRANSFORM_MAP[name])
    return transforms


def _build_f_mean(
    parent_names: list[str],
    coefs: dict[str, float],
    intercept: float,
    transforms: list[Callable[[Tensor], Tensor]] | None,
) -> Callable[[dict[str, Tensor]], Tensor]:
    """Build a mean function from linear combination and optional transforms."""

    # Gradient-standardized linear predictor:
    # normalize by Euclidean coefficient scale so random draw realizations keep
    # a controlled local sensitivity before nonlinear transforms.
    preact_scale = (
        (1.0 + sum(coefs[parent] ** 2 for parent in parent_names)) ** 0.5
        if parent_names
        else 1.0
    )

    def f_mean(parents: dict[str, Tensor]) -> Tensor:
        if not parent_names:
            base = torch.tensor(intercept)
        else:
            base = torch.tensor(intercept)
            for parent_name in parent_names:
                base = base + coefs[parent_name] * parents[parent_name]
            base = base / preact_scale

        if transforms:
            return _compose_transforms(base, transforms)
        return base

    return f_mean


def _sample_parents(
    rng: random.Random,
    previous_names: list[str],
    parent_prob: float,
) -> list[str]:
    """Sample parent names from previous variables (guarantees DAG)."""
    return [name for name in previous_names if rng.random() < parent_prob]


def _validate_dag(variables: list[str], parent_map: dict[str, list[str]]) -> None:
    """Validate that a parent map defines a DAG."""
    in_degree = {name: 0 for name in variables}
    children: dict[str, list[str]] = {name: [] for name in variables}
    for child, parents in parent_map.items():
        for parent in parents:
            if parent not in in_degree:
                raise ValueError(f"Unknown parent '{parent}' for '{child}'.")
            in_degree[child] += 1
            children[parent].append(child)

    queue = [name for name, degree in in_degree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(variables):
        raise ValueError("Parent assignment produced a cyclic graph.")


def random_scm(config: RandomSCMConfig) -> SCM:
    """Generate a random SCM with nonlinear mean functions and DAG structure."""
    if config.num_variables <= 0:
        raise ValueError("num_variables must be positive.")
    if not (0.0 <= config.parent_prob <= 1.0):
        raise ValueError("parent_prob must be between 0 and 1.")
    if not (0.0 <= config.categorical_prob <= 1.0):
        raise ValueError("categorical_prob must be between 0 and 1.")
    if config.num_categories_range[0] < 2:
        raise ValueError("num_categories_range minimum must be >= 2.")
    if config.num_categories_range[0] > config.num_categories_range[1]:
        raise ValueError("num_categories_range must satisfy min <= max.")

    rng = random.Random(config.seed)
    variable_names = [f"X{i}" for i in range(config.num_variables)]
    sampled_categorical_flags = [
        rng.random() < config.categorical_prob for _ in range(config.num_variables)
    ]

    if config.num_variables >= 2:
        if config.categorical_prob > 0.0 and not any(sampled_categorical_flags):
            sampled_categorical_flags[rng.randrange(config.num_variables)] = True
        if all(sampled_categorical_flags):
            sampled_categorical_flags[rng.randrange(config.num_variables)] = False
    if config.num_variables >= 3:
        while sum(not is_cat for is_cat in sampled_categorical_flags) < 2:
            categorical_indices = [
                idx for idx, is_cat in enumerate(sampled_categorical_flags) if is_cat
            ]
            if not categorical_indices:
                break
            sampled_categorical_flags[rng.choice(categorical_indices)] = False

    parent_map: dict[str, list[str]] = {}
    variables: list[Variable] = []
    variable_kinds: dict[str, Literal["gaussian", "categorical"]] = {}
    variable_categories: dict[str, int] = {}

    for idx, name in enumerate(variable_names):
        previous = variable_names[:idx]
        parents = _sample_parents(
            rng, previous_names=previous, parent_prob=config.parent_prob
        )
        parent_map[name] = parents

        sigma = rng.uniform(*config.sigma_range)
        intercept = rng.uniform(*config.intercept_range)
        if sampled_categorical_flags[idx]:
            num_categories = rng.randint(*config.num_categories_range)
            temperature = rng.uniform(*config.categorical_temperature_range)
            categorical_bias = [
                rng.uniform(*config.categorical_bias_range)
                for _ in range(num_categories)
            ]
            parent_kinds = {
                parent: variable_kinds[parent]
                for parent in parents
            }

            gaussian_parent_weights: dict[str, list[float]] = {}
            categorical_parent_weights: dict[str, list[list[float]]] = {}
            for parent in parents:
                if variable_kinds[parent] == "gaussian":
                    gaussian_parent_weights[parent] = [
                        rng.uniform(*config.categorical_weight_range)
                        for _ in range(num_categories)
                    ]
                else:
                    parent_categories = variable_categories[parent]
                    categorical_parent_weights[parent] = [
                        [
                            rng.uniform(*config.categorical_weight_range)
                            for _ in range(num_categories)
                        ]
                        for _ in range(parent_categories)
                    ]

            transform_names = (
                _sample_transform_names(rng) if rng.random() < config.nonlinear_prob else []
            )

            variables.append(
                RandomCategoricalVariable(
                    name=name,
                    parent_names=parents,
                    parent_kinds=parent_kinds,
                    num_categories=num_categories,
                    temperature=temperature,
                    bias=categorical_bias,
                    gaussian_parent_weights=gaussian_parent_weights,
                    categorical_parent_weights=categorical_parent_weights,
                    transforms=transform_names,
                )
            )
            variable_kinds[name] = "categorical"
            variable_categories[name] = num_categories
            continue

        use_nonlinear = rng.random() < config.nonlinear_prob
        transform_names = _sample_transform_names(rng) if use_nonlinear else None
        transforms = resolve_transforms(transform_names) if transform_names else None

        parent_projections: dict[str, list[float]] = {
            parent: [
                rng.uniform(*config.coef_range)
                for _ in range(variable_categories[parent])
            ]
            for parent in parents
            if variable_kinds.get(parent) == "categorical"
        }

        def _as_scalar(parent_name: str, parent_value: Tensor) -> Tensor:
            if parent_value.ndim <= 1:
                return parent_value
            projection = torch.tensor(
                parent_projections[parent_name],
                device=parent_value.device,
                dtype=parent_value.dtype,
            )
            return (parent_value * projection).sum(dim=-1)

        scalar_coefs = {parent: rng.uniform(*config.coef_range) for parent in parents}

        def mixed_parent_mean(base_parents: dict[str, Tensor]) -> Tensor:
            if not parents:
                base = torch.tensor(intercept)
            else:
                ref = next(iter(base_parents.values()))
                base = torch.zeros_like(ref[..., 0] if ref.ndim > 1 else ref) + intercept
                for parent_name in parents:
                    parent_scalar = _as_scalar(parent_name, base_parents[parent_name])
                    base = base + scalar_coefs[parent_name] * parent_scalar
            if transforms:
                return _compose_transforms(base, transforms)
            return base

        has_categorical_parent = any(
            variable_kinds.get(parent_name) == "categorical"
            for parent_name in parents
        )

        if use_nonlinear or has_categorical_parent:
            variables.append(
                FunctionalVariable(
                    name=name,
                    parent_names=parents,
                    sigma=sigma,
                    f_mean=mixed_parent_mean,
                    coefs=scalar_coefs,
                    intercept=intercept,
                    transforms=transform_names,
                )
            )
        else:
            variables.append(
                LinearVariable(
                    name=name,
                    parent_names=parents,
                    sigma=sigma,
                    coefs=scalar_coefs,
                    intercept=intercept,
                )
            )
        variable_kinds[name] = "gaussian"

    _validate_dag(variable_names, parent_map)

    return SCM(variables=variables)
