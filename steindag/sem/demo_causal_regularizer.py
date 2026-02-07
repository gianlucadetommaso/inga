"""Demo: train an MLP with and without causal regularization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig
from steindag.sem.regularizer import CausalRegularizer


@dataclass
class DemoConfig:
    """Configuration for the causal regularizer demo."""

    num_samples: int = 1000
    num_epochs: int = 200
    learning_rate: float = 1e-3
    regularizer_weight: float = 1.0
    seed: int = 0
    regularizer_batch: int = 128


def _build_mlp(input_dim: int) -> nn.Module:
    """Build a simple MLP for regression."""
    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def _train_model(
    model: nn.Module,
    observed: dict[str, Tensor],
    targets: Tensor,
    regularizer: CausalRegularizer | None,
    config: DemoConfig,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    for _ in range(config.num_epochs):
        optimizer.zero_grad()
        if regularizer is not None:
            inputs = regularizer.ravel_observed(observed)
        else:
            inputs = torch.stack([observed[name] for name in observed], dim=1)
        preds = model(inputs).squeeze(1)
        loss = loss_fn(preds, targets)

        if regularizer is not None:
            reg_observed = {
                name: values[: config.regularizer_batch]
                for name, values in observed.items()
            }
            reg_term = regularizer.regularization_term(
                reg_observed,
                num_samples=50,
                reduction="mean",
                enable_grad=True,
            )
            loss = loss + config.regularizer_weight * reg_term

        loss.backward()
        optimizer.step()

    return model


def _gradient_wrt_treatment(
    model: nn.Module,
    observed: dict[str, Tensor],
    observed_names: list[str],
    treatment_name: str,
) -> Tensor:
    inputs = torch.stack([observed[name] for name in observed_names], dim=1)
    inputs.requires_grad_(True)
    preds = model(inputs).squeeze(1)
    grads = torch.autograd.grad(preds.sum(), inputs, create_graph=False)[0]
    treatment_index = observed_names.index(treatment_name)
    return grads[:, treatment_index]


def run_demo(config: DemoConfig) -> None:
    torch.manual_seed(config.seed)

    dataset = None
    for attempt in range(5):
        seed = config.seed + attempt
        dataset_config = SEMDatasetConfig(
            sem_config=RandomSEMConfig(
                num_variables=6,
                parent_prob=0.6,
                nonlinear_prob=0.0,
                coef_range=(-1.0, 1.0),
                intercept_range=(-0.5, 0.5),
                seed=seed,
            ),
            num_samples=config.num_samples,
            num_queries=10,
            min_observed=1,
            seed=seed,
        )
        try:
            dataset = generate_sem_dataset(dataset_config)
            break
        except Exception as exc:  # pragma: no cover - demo-only retry logic
            if attempt == 4:
                raise exc

    if dataset is None:
        raise RuntimeError("Failed to generate SEM dataset for demo.")
    query = dataset.queries[0]
    observed = {name: dataset.data[name] for name in query.observed_names}
    targets = dataset.data[query.outcome_name]

    dataset.sem.posterior.fit(observed)

    input_dim = len(query.observed_names)

    model_plain = _build_mlp(input_dim)
    model_reg = _build_mlp(input_dim)

    regularizer = CausalRegularizer(
        sem=dataset.sem,
        model=model_reg,
        observed_names=query.observed_names,
        treatment_name=query.treatment_name,
        outcome_name=query.outcome_name,
    )

    _train_model(model_plain, observed, targets, None, config)
    _train_model(model_reg, observed, targets, regularizer, config)

    grad_plain = _gradient_wrt_treatment(
        model_plain, observed, query.observed_names, query.treatment_name
    )
    grad_reg = _gradient_wrt_treatment(
        model_reg, observed, query.observed_names, query.treatment_name
    )

    key = (
        query.treatment_name,
        query.outcome_name,
        tuple(query.observed_names),
    )
    true_effect = dataset.causal_effects[key]

    mae_plain = torch.mean(torch.abs(grad_plain - true_effect)).item()
    mae_reg = torch.mean(torch.abs(grad_reg - true_effect)).item()
    plain_effect_mean = grad_plain.mean().item()
    reg_effect_mean = grad_reg.mean().item()
    true_effect_mean = true_effect.mean().item()

    print("Causal regularizer demo")
    print(f"Treatment: {query.treatment_name} | Outcome: {query.outcome_name}")
    print(f"Observed: {', '.join(query.observed_names)}")
    print()
    headers = ["Model", "Causal effect (mean grad)", "Ground truth (mean)", "MAE"]
    rows = [
        [
            "Plain",
            f"{plain_effect_mean:.4f}",
            f"{true_effect_mean:.4f}",
            f"{mae_plain:.4f}",
        ],
        [
            "Regularized",
            f"{reg_effect_mean:.4f}",
            f"{true_effect_mean:.4f}",
            f"{mae_reg:.4f}",
        ],
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    def _fmt(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(_fmt(headers))
    print(sep)
    for row in rows:
        print(_fmt(row))
    print(sep)


if __name__ == "__main__":
    run_demo(DemoConfig())
