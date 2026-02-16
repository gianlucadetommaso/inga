"""Utilities shared by example/benchmark scripts.

This module intentionally has *no* dependency on inga internals, so it can
be reused across multiple example scripts.
"""

from __future__ import annotations

import statistics

import torch
from torch import Tensor
from collections import defaultdict


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    separator_after: set[int] | None = None,
) -> None:
    """Print an ASCII table.

    Args:
        separator_after: if provided, prints a separator line after each row
            index contained in the set.
    """

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt(row: list[str]) -> str:
        return (
            "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"
        )

    print(f"\n{title}")
    print(sep)
    print(fmt(headers))
    print(sep)
    for idx, row in enumerate(rows):
        print(fmt(row))
        if separator_after is not None and idx in separator_after:
            print(sep)
    print(sep)


def print_table_grouped(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    group_col: int,
) -> None:
    """Print an ASCII table and insert separator lines when a grouping key changes."""
    if not rows:
        print_table(title, headers, rows)
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    def fmt(row: list[str]) -> str:
        return (
            "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"
        )

    print(f"\n{title}")
    print(sep)
    print(fmt(headers))
    print(sep)

    prev_group = rows[0][group_col]
    for row in rows:
        group = row[group_col]
        if group != prev_group:
            print(sep)
            prev_group = group
        print(fmt(row))
    print(sep)


def pad_2d_with_mask(x: Tensor, max_features: int) -> tuple[Tensor, Tensor]:
    """Pad a 2D tensor (N,T) to (N,max_T) and return (padded, mask)."""
    n, t = x.shape
    mask = torch.zeros(n, max_features, dtype=torch.float32, device=x.device)
    mask[:, :t] = 1.0
    if t == max_features:
        return x, mask
    pad = torch.zeros(n, max_features - t, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1), mask


def pad_2d(x: Tensor, max_features: int) -> Tensor:
    """Pad a 2D tensor (N,T) to (N,max_T)."""
    n, t = x.shape
    if t == max_features:
        return x
    pad = torch.zeros(n, max_features - t, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def permute_columns(
    x: Tensor,
    mask: Tensor,
    *ys: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Apply the same random feature permutation to x/mask and any aligned targets.

    All tensors must be shaped (N, max_T).
    """
    max_t = x.shape[1]
    perm = torch.randperm(max_t, device=x.device, generator=generator)
    x_p = x[:, perm]
    mask_p = mask[:, perm]
    ys_p = tuple(y[:, perm] for y in ys)
    return x_p, mask_p, ys_p


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Compute mean absolute error over only the masked entries."""
    err = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return err.sum() / denom


def extract_observed_bundle(
    dataset: object,
    *,
    strategy: str = "max_treatments",
) -> tuple[list[str], str, Tensor, Tensor, Tensor]:
    """Extract a bundle (observed features, outcome, y, CE, CB) from a SCM dataset.

    This unifies extraction across benchmarks.

    Args:
        strategy:
            - "first": use dataset.queries[0]
            - "max_treatments": choose the outcome/observed bundle that yields the
              largest number of treatments (i.e. most observed features).

    Returns:
        feature_names: list[str] (observed variables)
        outcome_name: str
        y_all: (N,)
        ce_all: (N, T)
        cb_all: (N, T)
    """

    if not hasattr(dataset, "queries"):
        raise AttributeError("dataset must have a .queries attribute")
    if len(dataset.queries) == 0:
        raise ValueError("dataset.queries is empty")

    if strategy == "first":
        query = dataset.queries[0]
        outcome_name = query.outcome_name
        feature_names = list(query.observed_names)
    elif strategy == "max_treatments":
        groups: dict[tuple[str, tuple[str, ...]], list[object]] = defaultdict(list)
        for q in dataset.queries:
            groups[(q.outcome_name, tuple(q.observed_names))].append(q)
        (outcome_name, observed_tuple), _ = max(
            groups.items(), key=lambda kv: len(kv[1])
        )
        feature_names = list(observed_tuple)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ce_cols: list[Tensor] = []
    cb_cols: list[Tensor] = []
    for treatment_name in feature_names:
        key = (treatment_name, outcome_name, tuple(feature_names))
        ce_cols.append(dataset.causal_effects[key])
        cb_cols.append(dataset.causal_biases[key])

    y_all = dataset.data[outcome_name]
    ce_all = torch.stack(ce_cols, dim=1)
    cb_all = torch.stack(cb_cols, dim=1)
    return feature_names, outcome_name, y_all, ce_all, cb_all
