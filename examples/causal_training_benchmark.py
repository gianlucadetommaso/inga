"""Benchmark causal-aware training on random SEM datasets.

Compares three regimes over multiple random seeds:
1) standard MLP prediction,
2) L2-regularized MLP prediction,
3) causal multi-head model with prediction/effect/bias heads and
   causal-consistency regularization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig

from examples.utils import extract_observed_bundle, print_table, summary


@dataclass
class RegimeResult:
    pred_mae: float
    ce_mae: float


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class CausalThreeHead(nn.Module):
    def __init__(self, in_dim: int, num_treatments: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pred_head = nn.Linear(hidden_dim, 1)
        self.ce_head = nn.Linear(hidden_dim, num_treatments)
        self.cb_head = nn.Linear(hidden_dim, num_treatments)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(x)
        return (
            self.pred_head(h).squeeze(-1),
            self.ce_head(h),
            self.cb_head(h),
        )


def _estimate_effect_from_gradient(
    model: MLPRegressor,
    x: Tensor,
    treatment_indices: list[int],
) -> Tensor:
    x_grad = x.detach().clone().requires_grad_(True)
    pred = model(x_grad)
    grad = torch.autograd.grad(pred.sum(), x_grad, create_graph=False)[0]
    return grad[:, treatment_indices].detach()


def train_standard_or_l2(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    true_ce_test_by_treatment: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> RegimeResult:
    model = MLPRegressor(in_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test)
        pred_mae = (pred_test - y_test).abs().mean().item()
    ce_pred = _estimate_effect_from_gradient(
        model, x_test, treatment_indices=list(range(x_test.shape[1]))
    )
    ce_mae = (ce_pred - true_ce_test_by_treatment).abs().mean().item()
    return RegimeResult(pred_mae=pred_mae, ce_mae=ce_mae)


def train_causal_three_head(
    x_train: Tensor,
    y_train: Tensor,
    ce_train_by_treatment: Tensor,
    cb_train_by_treatment: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    true_ce_test_by_treatment: Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_ce: float,
    lambda_cb: float,
    lambda_consistency: float,
) -> RegimeResult:
    num_treatments = ce_train_by_treatment.shape[1]
    model = CausalThreeHead(in_dim=x_train.shape[1], num_treatments=num_treatments)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    dataset = TensorDataset(
        x_train, y_train, ce_train_by_treatment, cb_train_by_treatment
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb, ceb, cbb in loader:
            xb = xb.detach().clone().requires_grad_(True)
            optimizer.zero_grad()

            pred, ce_hat, cb_hat = model(xb)
            pred_loss = mse(pred, yb)
            ce_loss = l1(ce_hat, ceb)
            cb_loss = l1(cb_hat, cbb)

            grad_pred = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]
            consistency_loss = l1(grad_pred, ce_hat + cb_hat)

            loss = (
                pred_loss
                + lambda_ce * ce_loss
                + lambda_cb * cb_loss
                + lambda_consistency * consistency_loss
            )
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test, ce_test_hat, _ = model(x_test)
        pred_mae = (pred_test - y_test).abs().mean().item()
        ce_mae = (ce_test_hat - true_ce_test_by_treatment).abs().mean().item()
    return RegimeResult(pred_mae=pred_mae, ce_mae=ce_mae)


def run_seed(
    seed: int,
    *,
    train_size: int,
    test_size: int,
    query_samples: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, RegimeResult]:
    torch.manual_seed(seed)

    dataset = generate_sem_dataset(
        SEMDatasetConfig(
            sem_config=RandomSEMConfig(
                num_variables=6,
                parent_prob=0.6,
                nonlinear_prob=0.8,
                sigma_range=(0.7, 1.2),
                coef_range=(-1.0, 1.0),
                intercept_range=(-0.5, 0.5),
                seed=seed,
            ),
            num_samples=train_size + test_size,
            num_queries=1,
            min_observed=1,
            seed=seed,
        )
    )

    feature_names, _outcome_name, y_all, ce_all, cb_all = extract_observed_bundle(
        dataset, strategy="max_treatments"
    )
    x_all = torch.stack([dataset.data[name] for name in feature_names], dim=1)

    x_train_raw, x_test_raw = x_all[:train_size], x_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]
    ce_train, ce_test = ce_all[:train_size], ce_all[train_size:]
    cb_train = cb_all[:train_size]

    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std

    # query_samples kept as an arg for API symmetry / easy experimentation.
    _ = query_samples

    standard = train_standard_or_l2(
        x_train,
        y_train,
        x_test,
        y_test,
        ce_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=0.0,
    )
    l2 = train_standard_or_l2(
        x_train,
        y_train,
        x_test,
        y_test,
        ce_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-3,
    )
    causal = train_causal_three_head(
        x_train,
        y_train,
        ce_train,
        cb_train,
        x_test,
        y_test,
        ce_test,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lambda_ce=1.0,
        lambda_cb=1.0,
        lambda_consistency=1.0,
    )
    return {
        "standard": standard,
        "l2": l2,
        "causal_multitask": causal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--train-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--query-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    per_regime_pred: dict[str, list[float]] = {
        "standard": [],
        "l2": [],
        "causal_multitask": [],
    }
    per_regime_ce: dict[str, list[float]] = {
        "standard": [],
        "l2": [],
        "causal_multitask": [],
    }
    per_seed_rows: list[list[str]] = []

    for seed in range(args.num_seeds):
        result = run_seed(
            seed,
            train_size=args.train_size,
            test_size=args.test_size,
            query_samples=args.query_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        seed_rows: list[list[str]] = []
        for regime, metrics in result.items():
            per_regime_pred[regime].append(metrics.pred_mae)
            per_regime_ce[regime].append(metrics.ce_mae)
            row = [
                f"{seed:02d}",
                regime,
                f"{metrics.pred_mae:.6f}",
                f"{metrics.ce_mae:.6f}",
            ]
            per_seed_rows.append(row)
            seed_rows.append(row)

        print_table(
            title=f"Per-seed results (seed={seed:02d})",
            headers=[
                "seed",
                "regime",
                "prediction_mae",
                "causal_effect_mae_avg_over_treatments",
            ],
            rows=seed_rows,
        )

    regimes_per_seed = len(per_regime_pred)
    separator_after = {
        idx
        for idx in range(regimes_per_seed - 1, len(per_seed_rows), regimes_per_seed)
        if idx < len(per_seed_rows) - 1
    }

    print_table(
        "All per-seed results",
        [
            "seed",
            "regime",
            "prediction_mae",
            "causal_effect_mae_avg_over_treatments",
        ],
        per_seed_rows,
        separator_after=separator_after,
    )

    summary_rows: list[list[str]] = []
    for regime in per_regime_pred:
        pred_stats = summary(per_regime_pred[regime])
        ce_stats = summary(per_regime_ce[regime])
        summary_rows.append(
            [
                regime,
                f"{pred_stats['mean']:.6f}",
                f"{pred_stats['median']:.6f}",
                f"{pred_stats['std']:.6f}",
                f"{ce_stats['mean']:.6f}",
                f"{ce_stats['median']:.6f}",
                f"{ce_stats['std']:.6f}",
            ]
        )

    print_table(
        "Summary across seeds",
        [
            "regime",
            "pred_mean",
            "pred_median",
            "pred_std",
            "ce_mean",
            "ce_median",
            "ce_std",
        ],
        summary_rows,
    )


if __name__ == "__main__":
    main()
