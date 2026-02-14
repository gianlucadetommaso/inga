"""Random SEM generator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import random
import torch
from torch import Tensor
from torch.nn import functional as F

from steindag.sem.base import SEM
from steindag.variable.base import Variable
from steindag.variable.linear import LinearVariable
from steindag.variable.functional import FunctionalVariable


@dataclass(frozen=True)
class RandomSEMConfig:
    """Configuration for random SEM generation."""

    num_variables: int
    parent_prob: float = 0.3
    sigma_range: tuple[float, float] = (0.5, 1.5)
    coef_range: tuple[float, float] = (-2.0, 2.0)
    intercept_range: tuple[float, float] = (-1.0, 1.0)
    nonlinear_prob: float = 0.5
    seed: int | None = None


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


def random_sem(config: RandomSEMConfig) -> SEM:
    """Generate a random SEM with nonlinear mean functions and DAG structure."""
    if config.num_variables <= 0:
        raise ValueError("num_variables must be positive.")
    if not (0.0 <= config.parent_prob <= 1.0):
        raise ValueError("parent_prob must be between 0 and 1.")

    rng = random.Random(config.seed)
    variable_names = [f"X{i}" for i in range(config.num_variables)]
    parent_map: dict[str, list[str]] = {}
    variables: list[Variable] = []

    for idx, name in enumerate(variable_names):
        previous = variable_names[:idx]
        parents = _sample_parents(
            rng, previous_names=previous, parent_prob=config.parent_prob
        )
        parent_map[name] = parents

        sigma = rng.uniform(*config.sigma_range)
        intercept = rng.uniform(*config.intercept_range)
        coefs = {parent: rng.uniform(*config.coef_range) for parent in parents}

        use_nonlinear = rng.random() < config.nonlinear_prob
        transform_names = _sample_transform_names(rng) if use_nonlinear else None
        transforms = resolve_transforms(transform_names) if transform_names else None

        if use_nonlinear:
            f_mean = _build_f_mean(parents, coefs, intercept, transforms)
            variables.append(
                FunctionalVariable(
                    name=name,
                    parent_names=parents,
                    sigma=sigma,
                    f_mean=f_mean,
                    coefs=coefs,
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
                    coefs=coefs,
                    intercept=intercept,
                )
            )

    _validate_dag(variable_names, parent_map)
    return SEM(variables=variables)
