"""Compare two causal-training methods on one random SEM dataset.

Method 1 (effect-gradient):
    Train a predictor with a loss that matches d y_hat / d x to causal effect.

Method 2 (three-head structure):
    Train heads for y, effect, bias with consistency:
        d y_hat / d x ≈ effect_hat + bias_hat.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig


def _make_mlp(hidden_dim: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


class ThreeHeadModel(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.pred_head = nn.Linear(hidden_dim, 1)
        self.effect_head = nn.Linear(hidden_dim, 1)
        self.bias_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(x)
        return self.pred_head(h), self.effect_head(h), self.bias_head(h)


def _estimate_effect_from_predictor(model: nn.Module, x: Tensor) -> Tensor:
    x_req = x.clone().detach().requires_grad_(True)
    y_pred = model(x_req.unsqueeze(1)).squeeze(1)
    return torch.autograd.grad(y_pred.sum(), x_req)[0].detach()


def _train_baseline(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_in = x.unsqueeze(1)
    y_t = y.unsqueeze(1)

    for _ in range(epochs):
        pred = model(x_in)
        loss = torch.mean((pred - y_t) ** 2)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    return model


def _train_method1_effect_gradient(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    effect_target: Tensor,
    epochs: int,
    lr: float,
    lambda_effect: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_base = x.unsqueeze(1)
    y_t = y.unsqueeze(1)
    effect_t = effect_target.unsqueeze(1)
    scale = torch.mean(torch.abs(effect_t)).detach() + 1e-6

    for epoch in range(epochs):
        warmup = min(1.0, float(epoch + 1) / max(1, epochs // 5))
        x_in = x_base.clone().detach().requires_grad_(True)
        pred = model(x_in)
        dy_dx = torch.autograd.grad(pred.sum(), x_in, create_graph=True)[0]

        pred_loss = torch.mean((pred - y_t) ** 2)
        effect_loss = F.smooth_l1_loss((dy_dx - effect_t) / scale, torch.zeros_like(effect_t))
        loss = pred_loss + warmup * lambda_effect * effect_loss
        if not torch.isfinite(loss):
            continue

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


def _train_method2_three_head(
    model: ThreeHeadModel,
    x: Tensor,
    y: Tensor,
    effect: Tensor,
    bias: Tensor,
    epochs: int,
    lr: float,
    lambda_effect: float,
    lambda_bias: float,
    lambda_consistency: float,
) -> ThreeHeadModel:
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_base = x.unsqueeze(1)
    y_t = y.unsqueeze(1)
    effect_t = effect.unsqueeze(1)
    bias_t = bias.unsqueeze(1)
    eff_scale = torch.mean(torch.abs(effect_t)).detach() + 1e-6
    bias_scale = torch.mean(torch.abs(bias_t)).detach() + 1e-6

    for epoch in range(epochs):
        warmup = min(1.0, float(epoch + 1) / max(1, epochs // 5))
        x_in = x_base.clone().detach().requires_grad_(True)
        y_hat, e_hat, b_hat = model(x_in)
        dy_dx = torch.autograd.grad(y_hat.sum(), x_in, create_graph=True)[0]

        pred_loss = torch.mean((y_hat - y_t) ** 2)
        effect_loss = F.smooth_l1_loss((e_hat - effect_t) / eff_scale, torch.zeros_like(effect_t))
        bias_loss = F.smooth_l1_loss((b_hat - bias_t) / bias_scale, torch.zeros_like(bias_t))

        cons_res = dy_dx - (e_hat + b_hat)
        cons_scale = torch.mean(torch.abs(effect_t + bias_t)).detach() + 1e-6
        cons_loss = F.smooth_l1_loss(cons_res / cons_scale, torch.zeros_like(cons_res))

        loss = pred_loss + warmup * (
            lambda_effect * effect_loss + lambda_bias * bias_loss + lambda_consistency * cons_loss
        )
        if not torch.isfinite(loss):
            continue

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    return model


def run_demo(
    seed: int = 0,
    train_epochs: int = 300,
    lr: float = 1e-3,
    lambda_effect_m1: float = 1.0,
    lambda_effect_m2: float = 0.5,
    lambda_bias_m2: float = 0.5,
    lambda_consistency_m2: float = 1.0,
    verbose: bool = True,
) -> dict[str, float]:
    torch.manual_seed(seed)

    dataset = generate_sem_dataset(
        SEMDatasetConfig(
            sem_config=RandomSEMConfig(
                num_variables=6,
                parent_prob=0.65,
                nonlinear_prob=1.0,
            ),
            num_samples=1200,
            num_queries=1,
            min_observed=1,
            seed=seed,
        )
    )

    query = dataset.queries[0]
    key = (query.treatment_name, query.outcome_name, tuple(query.observed_names))
    x = dataset.data[query.treatment_name]
    y = dataset.data[query.outcome_name]
    effect = dataset.causal_effects[key]
    bias = dataset.causal_biases[key]

    n = x.shape[0]
    perm = torch.randperm(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]

    x_train, y_train = x[tr], y[tr]
    effect_train, bias_train = effect[tr], bias[tr]
    x_test, y_test = x[te], y[te]
    effect_test, bias_test = effect[te], bias[te]

    # Baseline (prediction-only)
    baseline = _train_baseline(
        _make_mlp(),
        x_train,
        y_train,
        epochs=train_epochs,
        lr=lr,
    )
    yhat_base = baseline(x_test.unsqueeze(1)).squeeze(1).detach()
    effect_base = torch.nan_to_num(_estimate_effect_from_predictor(baseline, x_test))
    bias_base = torch.zeros_like(effect_base)

    base_pred_mae = torch.mean(torch.abs(yhat_base - y_test)).item()
    base_effect_mae = torch.mean(torch.abs(effect_base - effect_test)).item()
    base_bias_mae = torch.mean(torch.abs(bias_base - bias_test)).item()
    base_consistency = 0.0

    # Method 1
    m1 = _train_method1_effect_gradient(
        _make_mlp(),
        x_train,
        y_train,
        effect_train,
        epochs=train_epochs,
        lr=lr,
        lambda_effect=lambda_effect_m1,
    )
    yhat_m1 = m1(x_test.unsqueeze(1)).squeeze(1).detach()
    effect_m1 = torch.nan_to_num(_estimate_effect_from_predictor(m1, x_test))
    bias_m1 = torch.zeros_like(effect_m1)

    m1_pred_mae = torch.mean(torch.abs(yhat_m1 - y_test)).item()
    m1_effect_mae = torch.mean(torch.abs(effect_m1 - effect_test)).item()
    m1_bias_mae = torch.mean(torch.abs(bias_m1 - bias_test)).item()
    m1_consistency = 0.0  # by construction: effect_m1 + bias_m1 == dy/dx

    # Method 2
    m2 = _train_method2_three_head(
        ThreeHeadModel(),
        x_train,
        y_train,
        effect_train,
        bias_train,
        epochs=train_epochs,
        lr=lr,
        lambda_effect=lambda_effect_m2,
        lambda_bias=lambda_bias_m2,
        lambda_consistency=lambda_consistency_m2,
    )

    x_req = x_test.clone().detach().requires_grad_(True)
    yhat2, ehat2, bhat2 = m2(x_req.unsqueeze(1))
    dydx2 = torch.autograd.grad(yhat2.sum(), x_req)[0].detach()
    yhat2 = torch.nan_to_num(yhat2.squeeze(1).detach())
    ehat2 = torch.nan_to_num(ehat2.squeeze(1).detach())
    bhat2 = torch.nan_to_num(bhat2.squeeze(1).detach())
    dydx2 = torch.nan_to_num(dydx2)

    m2_pred_mae = torch.mean(torch.abs(yhat2 - y_test)).item()
    m2_effect_mae = torch.mean(torch.abs(ehat2 - effect_test)).item()
    m2_bias_mae = torch.mean(torch.abs(bhat2 - bias_test)).item()
    m2_consistency = torch.mean(torch.abs(dydx2 - (ehat2 + bhat2))).item()

    if verbose:
        print("=== Compare Causal Methods (single seed) ===")
        print(
            f"Treatment={query.treatment_name} | Outcome={query.outcome_name} | "
            f"Observed={query.observed_names}"
        )
        print("+-----------------------+-----------------+-----------------+-----------------+-----------------+")
        print("| method                | pred_mae_test   | effect_mae_test | bias_mae_test   | consistency_mae |")
        print("+-----------------------+-----------------+-----------------+-----------------+-----------------+")
        print(
            f"| baseline              | {base_pred_mae:>15.6f} | {base_effect_mae:>15.6f} |"
            f" {base_bias_mae:>15.6f} | {base_consistency:>15.6f} |"
        )
        print(
            f"| method1_effect_grad   | {m1_pred_mae:>15.6f} | {m1_effect_mae:>15.6f} |"
            f" {m1_bias_mae:>15.6f} | {m1_consistency:>15.6f} |"
        )
        print(
            f"| method2_three_head    | {m2_pred_mae:>15.6f} | {m2_effect_mae:>15.6f} |"
            f" {m2_bias_mae:>15.6f} | {m2_consistency:>15.6f} |"
        )
        print("+-----------------------+-----------------+-----------------+-----------------+-----------------+")

    return {
        "baseline_pred_mae": base_pred_mae,
        "baseline_effect_mae": base_effect_mae,
        "baseline_bias_mae": base_bias_mae,
        "baseline_consistency_mae": base_consistency,
        "method1_pred_mae": m1_pred_mae,
        "method1_effect_mae": m1_effect_mae,
        "method1_bias_mae": m1_bias_mae,
        "method1_consistency_mae": m1_consistency,
        "method2_pred_mae": m2_pred_mae,
        "method2_effect_mae": m2_effect_mae,
        "method2_bias_mae": m2_bias_mae,
        "method2_consistency_mae": m2_consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    run_demo(seed=args.seed, train_epochs=args.epochs, verbose=True)


if __name__ == "__main__":
    main()
