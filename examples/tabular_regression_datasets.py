"""Small utilities to download and load open-source tabular *regression* datasets.

This intentionally avoids extra dependencies (no pandas / sklearn): we rely on
urllib + numpy.

The goal is not to provide a comprehensive dataset zoo, but a stable handful of
datasets for the synthetic->real transfer benchmark.
"""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RegressionDataset:
    name: str
    X: np.ndarray  # (N, D)
    y: np.ndarray  # (N,)
    feature_names: list[str]
    target_name: str


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with urllib.request.urlopen(url) as r:  # noqa: S310 (example-only)
        data = r.read()
    path.write_bytes(data)


def _load_wine_quality(cache_dir: Path, *, variant: str) -> RegressionDataset:
    if variant not in {"red", "white"}:
        raise ValueError("variant must be 'red' or 'white'")

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
        f"winequality-{variant}.csv"
    )
    local = cache_dir / f"winequality-{variant}.csv"
    _download(url, local)

    # ';' delimited, header row.
    data = np.genfromtxt(local, delimiter=";", names=True, dtype=np.float64)
    feature_names = [
        name for name in data.dtype.names if name is not None and name != "quality"
    ]
    X = np.stack([data[name] for name in feature_names], axis=1)
    y = np.asarray(data["quality"], dtype=np.float64)
    return RegressionDataset(
        name=f"wine_quality_{variant}",
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="quality",
    )


def _load_airfoil(cache_dir: Path) -> RegressionDataset:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/"
        "airfoil_self_noise.dat"
    )
    local = cache_dir / "airfoil_self_noise.dat"
    _download(url, local)

    # Whitespace-delimited, 6 columns, no header.
    raw = np.loadtxt(local, dtype=np.float64)
    X = raw[:, :5]
    y = raw[:, 5]
    feature_names = [
        "frequency",
        "angle_of_attack",
        "chord_length",
        "free_stream_velocity",
        "suction_side_displacement_thickness",
    ]
    return RegressionDataset(
        name="airfoil_self_noise",
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="scaled_sound_pressure_level",
    )


def _load_yacht(cache_dir: Path) -> RegressionDataset:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/"
        "yacht_hydrodynamics.data"
    )
    local = cache_dir / "yacht_hydrodynamics.data"
    _download(url, local)

    # Whitespace-delimited, 7 columns, no header.
    raw = np.loadtxt(local, dtype=np.float64)
    X = raw[:, :6]
    y = raw[:, 6]
    feature_names = [
        "longitudinal_position",
        "prismatic_coefficient",
        "length_displacement_ratio",
        "beam_draught_ratio",
        "length_beam_ratio",
        "froude_number",
    ]
    return RegressionDataset(
        name="yacht_hydrodynamics",
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="residuary_resistance",
    )


def _load_abalone(cache_dir: Path) -> RegressionDataset:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/"
        "abalone.data"
    )
    local = cache_dir / "abalone.data"
    _download(url, local)

    # CSV without header. First col is categorical sex: M/F/I.
    rows: list[list[str]] = []
    with local.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    sex = np.array([r[0] for r in rows])
    sex_M = (sex == "M").astype(np.float64)
    sex_F = (sex == "F").astype(np.float64)
    sex_I = (sex == "I").astype(np.float64)

    numeric = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64)
    # numeric contains 7 predictors + target rings at last position.
    X_num = numeric[:, :7]
    y = numeric[:, 7]
    X = np.concatenate([sex_M[:, None], sex_F[:, None], sex_I[:, None], X_num], axis=1)
    feature_names = [
        "sex_M",
        "sex_F",
        "sex_I",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
    ]
    return RegressionDataset(
        name="abalone",
        X=X,
        y=y,
        feature_names=feature_names,
        target_name="rings",
    )


def load_regression_datasets(
    *,
    cache_dir: str | Path = ".data/tabular_regression",
    names: list[str] | None = None,
) -> list[RegressionDataset]:
    """Load a set of regression datasets, downloading them if missing.

    Args:
        names: optional subset. Supported names:
          - abalone
          - airfoil_self_noise
          - yacht_hydrodynamics
          - wine_quality_red
          - wine_quality_white
    """

    cache = Path(cache_dir)
    supported = {
        "abalone": lambda: _load_abalone(cache),
        "airfoil_self_noise": lambda: _load_airfoil(cache),
        "yacht_hydrodynamics": lambda: _load_yacht(cache),
        "wine_quality_red": lambda: _load_wine_quality(cache, variant="red"),
        "wine_quality_white": lambda: _load_wine_quality(cache, variant="white"),
    }

    selected = list(supported.keys()) if names is None else names
    datasets: list[RegressionDataset] = []
    for name in selected:
        if name not in supported:
            raise ValueError(
                f"Unknown dataset '{name}'. Supported: {sorted(supported.keys())}"
            )
        datasets.append(supported[name]())
    return datasets
