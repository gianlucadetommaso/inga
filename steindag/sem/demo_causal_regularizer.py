"""Demo: baseline vs L2 vs causal-regularized MLP using SEM dataset generation.

This demo uses `generate_sem_dataset`, which internally samples a random
nonlinear SEM and computes query-specific ground-truth quantities, including:

- causal effects
- causal regularization terms

We train three models on the same treatment/outcome query:
1) baseline (prediction loss only),
2) L2-regularized,
3) causal-regularized with |model(x) - regularization(x)|.

Finally, we estimate causal effect from each model as d model(x) / d x and
compare against dataset ground truth.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig


def _make_mlp(hidden_dim: int = 32) -> nn.Module:
    """Build a small MLP mapping treatment x to outcome y."""
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
    regularization_target: Tensor,
    lambda_reg: float,
    lambda_l2: float = 0.0,
    epochs: int = 500,
    lr: float = 1e-3,
) -> nn.Module:
    """Train model with optional causal and L2 penalties."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_in = x.unsqueeze(1)
    y_t = y.unsqueeze(1)
    reg_t = regularization_target.detach().unsqueeze(1)

    for _ in range(epochs):
        pred = model(x_in)
        pred_loss = torch.mean((pred - y_t) ** 2)
        reg_loss = torch.mean(torch.abs(pred - reg_t))

        l2_loss = torch.zeros((), device=pred.device)
        for param in model.parameters():
            l2_loss = l2_loss + torch.sum(param**2)

        loss = pred_loss + lambda_reg * reg_loss + lambda_l2 * l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def _estimate_causal_effect(model: nn.Module, x: Tensor) -> Tensor:
    """Estimate causal effect as d model(x) / d x for each sample."""
    x_req = x.clone().detach().requires_grad_(True)
    y_pred = model(x_req.unsqueeze(1)).squeeze(1)
    grad_x = torch.autograd.grad(y_pred.sum(), x_req)[0]
    return grad_x.detach()


def _print_results_table(
    treatment_name: str,
    outcome_name: str,
    observed_names: list[str],
    effect_true: Tensor,
    effect_baseline: Tensor,
    effect_l2: Tensor,
    effect_causal: Tensor,
) -> None:
    """Print a compact table with effect means and MAE vs ground truth."""
    rows = [
        ("ground_truth", effect_true.mean().item(), 0.0),
        (
            "baseline",
            effect_baseline.mean().item(),
            torch.mean(torch.abs(effect_baseline - effect_true)).item(),
        ),
        (
            "l2_regularized",
            effect_l2.mean().item(),
            torch.mean(torch.abs(effect_l2 - effect_true)).item(),
        ),
        (
            "causal_regularized",
            effect_causal.mean().item(),
            torch.mean(torch.abs(effect_causal - effect_true)).item(),
        ),
    ]

    header = (
        f"=== Causal Effect Estimation Demo (dataset-based) ===\n"
        f"Treatment={treatment_name} | Outcome={outcome_name} | Observed={observed_names}"
    )
    print(header)
    print("+--------------------+-------------+---------------------+")
    print("| model              | effect_mean | mae_vs_ground_truth |")
    print("+--------------------+-------------+---------------------+")
    for model_name, effect_mean, mae in rows:
        print(f"| {model_name:<18} | {effect_mean:>11.3f} | {mae:>19.4f} |")
    print("+--------------------+-------------+---------------------+")


def main() -> None:
    """Run demo and print causal-effect MAE comparison."""
    demo_seed = 43
    torch.manual_seed(demo_seed)

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
            seed=demo_seed,
        )
    )

    query = dataset.queries[0]
    key = (query.treatment_name, query.outcome_name, tuple(query.observed_names))

    x = dataset.data[query.treatment_name]
    y = dataset.data[query.outcome_name]
    reg_target = dataset.causal_regularizations[key]
    effect_true = dataset.causal_effects[key]

    num_samples = x.shape[0]
    perm = torch.randperm(num_samples)
    split = int(0.8 * num_samples)
    train_idx = perm[:split]
    test_idx = perm[split:]

    x_train, y_train, reg_train = x[train_idx], y[train_idx], reg_target[train_idx]
    x_test, effect_test = x[test_idx], effect_true[test_idx]

    baseline = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        regularization_target=reg_train,
        lambda_reg=0.0,
        lambda_l2=0.0,
    )

    l2_regularized = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        regularization_target=reg_train,
        lambda_reg=0.0,
        lambda_l2=1e-4,
    )

    causal_regularized = _train_model(
        model=_make_mlp(),
        x=x_train,
        y=y_train,
        regularization_target=reg_train,
        lambda_reg=0.1,
        lambda_l2=0.0,
    )

    effect_baseline = _estimate_causal_effect(baseline, x_test)
    effect_l2 = _estimate_causal_effect(l2_regularized, x_test)
    effect_causal = _estimate_causal_effect(causal_regularized, x_test)

    _print_results_table(
        treatment_name=query.treatment_name,
        outcome_name=query.outcome_name,
        observed_names=query.observed_names,
        effect_true=effect_test,
        effect_baseline=effect_baseline,
        effect_l2=effect_l2,
        effect_causal=effect_causal,
    )


if __name__ == "__main__":
    main()
