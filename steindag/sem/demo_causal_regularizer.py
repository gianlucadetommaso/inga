"""Demo: baseline vs L2 vs causal-effect-regularized MLP."""

from __future__ import annotations

import argparse

import torch
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


def _train_model(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    effect_target: Tensor | None,
    lambda_effect: float = 0.0,
    lambda_l2: float = 0.0,
    warmup_epochs: int = 1,
    epochs: int = 500,
    lr: float = 1e-3,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_base = x.unsqueeze(1)
    y_t = y.unsqueeze(1)
    effect_t = None if effect_target is None else effect_target.detach().unsqueeze(1)

    for epoch in range(epochs):
        use_effect_loss = effect_t is not None and lambda_effect > 0
        x_in = x_base.clone().detach().requires_grad_(True) if use_effect_loss else x_base

        warmup = min(1.0, float(epoch + 1) / max(1, warmup_epochs))
        pred = model(x_in)
        pred_loss = torch.mean((pred - y_t) ** 2)

        effect_loss = torch.zeros((), device=pred.device)
        if use_effect_loss:
            effect_hat = torch.autograd.grad(pred.sum(), x_in, create_graph=True)[0]
            effect_scale = torch.mean(torch.abs(effect_t)).detach() + 1e-6
            effect_residual = torch.nan_to_num((effect_hat - effect_t) / effect_scale)
            effect_loss = torch.mean(torch.abs(effect_residual))

        l2_loss = torch.zeros((), device=pred.device)
        for param in model.parameters():
            l2_loss = l2_loss + torch.sum(param**2)

        loss = pred_loss + (warmup * lambda_effect) * effect_loss + lambda_l2 * l2_loss
        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return model


def _estimate_causal_effect(model: nn.Module, x: Tensor) -> Tensor:
    x_req = x.clone().detach().requires_grad_(True)
    y_pred = model(x_req.unsqueeze(1)).squeeze(1)
    return torch.autograd.grad(y_pred.sum(), x_req)[0].detach()


def _print_results_table(
    treatment_name: str,
    outcome_name: str,
    observed_names: list[str],
    effect_true: Tensor,
    y_true: Tensor,
    effect_baseline: Tensor,
    effect_l2: Tensor,
    effect_causal: Tensor,
    yhat_baseline: Tensor,
    yhat_l2: Tensor,
    yhat_causal: Tensor,
) -> None:
    rows = [
        ("ground_truth", effect_true.mean().item(), 0.0, 0.0),
        (
            "baseline",
            effect_baseline.mean().item(),
            torch.mean(torch.abs(effect_baseline - effect_true)).item(),
            torch.mean(torch.abs(yhat_baseline - y_true)).item(),
        ),
        (
            "l2_regularized",
            effect_l2.mean().item(),
            torch.mean(torch.abs(effect_l2 - effect_true)).item(),
            torch.mean(torch.abs(yhat_l2 - y_true)).item(),
        ),
        (
            "causal_regularized",
            effect_causal.mean().item(),
            torch.mean(torch.abs(effect_causal - effect_true)).item(),
            torch.mean(torch.abs(yhat_causal - y_true)).item(),
        ),
    ]

    print("=== Causal Effect Estimation Demo ===")
    print(
        f"Treatment={treatment_name} | Outcome={outcome_name} | "
        f"Observed={observed_names}"
    )
    print("+--------------------+-------------+---------------------+-----------------+")
    print("| model              | effect_mean | effect_mae_on_test  | pred_mae_on_test|")
    print("+--------------------+-------------+---------------------+-----------------+")
    for model_name, effect_mean, effect_mae, pred_mae in rows:
        print(
            f"| {model_name:<18} | {effect_mean:>11.3f} | {effect_mae:>19.4f} |"
            f" {pred_mae:>15.4f} |"
        )
    print("+--------------------+-------------+---------------------+-----------------+")


def run_demo(seed: int = 28, verbose: bool = True, train_epochs: int = 500) -> dict[str, float]:
    """Run baseline/L2/causal-effect demo and return held-out effect MAEs."""
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
    effect_true = dataset.causal_effects[key]

    num_samples = x.shape[0]
    perm = torch.randperm(num_samples)
    split = int(0.8 * num_samples)
    train_idx = perm[:split]
    test_idx = perm[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    effect_train = effect_true[train_idx]
    x_test, effect_test = x[test_idx], effect_true[test_idx]
    y_test = y[test_idx]

    baseline = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        effect_target=None,
        lambda_effect=0.0,
        lambda_l2=0.0,
        epochs=train_epochs,
    )
    l2_regularized = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        effect_target=None,
        lambda_effect=0.0,
        lambda_l2=1e-4,
        epochs=train_epochs,
    )
    causal_regularized = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        effect_target=effect_train,
        lambda_effect=1.0,
        lambda_l2=0.0,
        warmup_epochs=max(20, train_epochs // 5),
        epochs=train_epochs,
    )

    effect_baseline = _estimate_causal_effect(baseline, x_test)
    effect_l2 = _estimate_causal_effect(l2_regularized, x_test)
    effect_causal = _estimate_causal_effect(causal_regularized, x_test)
    yhat_baseline = baseline(x_test.unsqueeze(1)).squeeze(1).detach()
    yhat_l2 = l2_regularized(x_test.unsqueeze(1)).squeeze(1).detach()
    yhat_causal = causal_regularized(x_test.unsqueeze(1)).squeeze(1).detach()

    mae_baseline = torch.mean(torch.abs(effect_baseline - effect_test)).item()
    mae_l2 = torch.mean(torch.abs(effect_l2 - effect_test)).item()
    mae_causal = torch.mean(torch.abs(effect_causal - effect_test)).item()
    pred_mae_baseline = torch.mean(torch.abs(yhat_baseline - y_test)).item()
    pred_mae_l2 = torch.mean(torch.abs(yhat_l2 - y_test)).item()
    pred_mae_causal = torch.mean(torch.abs(yhat_causal - y_test)).item()

    if verbose:
        _print_results_table(
            treatment_name=query.treatment_name,
            outcome_name=query.outcome_name,
            observed_names=query.observed_names,
            effect_true=effect_test,
            y_true=y_test,
            effect_baseline=effect_baseline,
            effect_l2=effect_l2,
            effect_causal=effect_causal,
            yhat_baseline=yhat_baseline,
            yhat_l2=yhat_l2,
            yhat_causal=yhat_causal,
        )

    return {
        "baseline": mae_baseline,
        "l2_regularized": mae_l2,
        "causal_regularized": mae_causal,
        "baseline_pred_mae": pred_mae_baseline,
        "l2_regularized_pred_mae": pred_mae_l2,
        "causal_regularized_pred_mae": pred_mae_causal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    run_demo(seed=args.seed, verbose=True, train_epochs=args.epochs)


if __name__ == "__main__":
    main()
