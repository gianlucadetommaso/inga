"""Demo: prediction-only model vs 3-head causal-structure model.

The structured model has three heads that predict:
1) outcome y,
2) causal effect,
3) causal bias.

In addition to supervised losses for those heads, it enforces:

    d y_hat / d x  ≈  effect_hat + bias_hat

to couple prediction behavior with the causal heads.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig


def _make_baseline(hidden_dim: int = 32) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


class ThreeHeadModel(nn.Module):
    """Shared trunk + three heads: prediction/effect/bias."""

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


def _train_baseline(model: nn.Module, x: Tensor, y: Tensor, epochs: int, lr: float) -> nn.Module:
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


def _train_structured(
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

    effect_scale = torch.mean(torch.abs(effect_t)).detach() + 1e-6
    bias_scale = torch.mean(torch.abs(bias_t)).detach() + 1e-6

    for epoch in range(epochs):
        warmup = min(1.0, float(epoch + 1) / max(1, epochs // 5))
        x_in = x_base.clone().detach().requires_grad_(True)

        y_hat, effect_hat, bias_hat = model(x_in)
        dy_dx = torch.autograd.grad(y_hat.sum(), x_in, create_graph=True)[0]

        pred_loss = torch.mean((y_hat - y_t) ** 2)
        effect_loss = F.smooth_l1_loss((effect_hat - effect_t) / effect_scale, torch.zeros_like(effect_t))
        bias_loss = F.smooth_l1_loss((bias_hat - bias_t) / bias_scale, torch.zeros_like(bias_t))

        consistency_residual = dy_dx - (effect_hat + bias_hat)
        consistency_scale = torch.mean(torch.abs(effect_t + bias_t)).detach() + 1e-6
        consistency_loss = F.smooth_l1_loss(
            consistency_residual / consistency_scale,
            torch.zeros_like(consistency_residual),
        )

        loss = (
            pred_loss
            + warmup * lambda_effect * effect_loss
            + warmup * lambda_bias * bias_loss
            + warmup * lambda_consistency * consistency_loss
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
    train_epochs: int = 400,
    lr: float = 1e-3,
    lambda_effect: float = 0.5,
    lambda_bias: float = 0.5,
    lambda_consistency: float = 1.0,
    verbose: bool = True,
) -> dict[str, float]:
    """Run one seed and compare test prediction MAE (baseline vs structured)."""
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

    baseline = _train_baseline(_make_baseline(), x_train, y_train, epochs=train_epochs, lr=lr)
    structured = _train_structured(
        ThreeHeadModel(),
        x_train,
        y_train,
        effect_train,
        bias_train,
        epochs=train_epochs,
        lr=lr,
        lambda_effect=lambda_effect,
        lambda_bias=lambda_bias,
        lambda_consistency=lambda_consistency,
    )

    # Baseline prediction + derived causal metrics.
    x_req_base = x_test.clone().detach().requires_grad_(True)
    yhat_base_graph = baseline(x_req_base.unsqueeze(1)).squeeze(1)
    dydx_base = torch.autograd.grad(yhat_base_graph.sum(), x_req_base)[0].detach()
    yhat_base = yhat_base_graph.detach()
    dydx_base = torch.nan_to_num(dydx_base)

    pred_mae_base = torch.mean(torch.abs(yhat_base - y_test)).item()
    effect_mae_base = torch.mean(torch.abs(dydx_base - effect_test)).item()
    bias_mae_base = torch.mean(torch.abs(torch.zeros_like(bias_test) - bias_test)).item()
    consistency_mae_base = 0.0

    # Structured prediction + structure metrics.
    x_req = x_test.clone().detach().requires_grad_(True)
    yhat_struct, ehat_struct, bhat_struct = structured(x_req.unsqueeze(1))
    dy_dx_struct = torch.autograd.grad(yhat_struct.sum(), x_req)[0].detach()
    yhat_struct = torch.nan_to_num(yhat_struct.squeeze(1).detach())
    ehat_struct = torch.nan_to_num(ehat_struct.squeeze(1).detach())
    bhat_struct = torch.nan_to_num(bhat_struct.squeeze(1).detach())
    dy_dx_struct = torch.nan_to_num(dy_dx_struct)

    pred_mae_struct = torch.mean(torch.abs(yhat_struct - y_test)).item()
    effect_head_mae = torch.mean(torch.abs(ehat_struct - effect_test)).item()
    bias_head_mae = torch.mean(torch.abs(bhat_struct - bias_test)).item()
    consistency_mae = torch.mean(torch.abs(dy_dx_struct - (ehat_struct + bhat_struct))).item()

    if verbose:
        print("=== Three-Head Causal-Structure Demo ===")
        print(
            f"Treatment={query.treatment_name} | Outcome={query.outcome_name} | "
            f"Observed={query.observed_names}"
        )
        print("+------------+-----------------+-----------------+-----------------+-----------------+")
        print("| model      | pred_mae_test   | effect_mae_test | bias_mae_test   | consistency_mae |")
        print("+------------+-----------------+-----------------+-----------------+-----------------+")
        print(
            f"| baseline   | {pred_mae_base:>15.6f} | {effect_mae_base:>15.6f} |"
            f" {bias_mae_base:>15.6f} | {consistency_mae_base:>15.6f} |"
        )
        print(
            f"| structured | {pred_mae_struct:>15.6f} | {effect_head_mae:>15.6f} |"
            f" {bias_head_mae:>15.6f} | {consistency_mae:>15.6f} |"
        )
        print("+------------+-----------------+-----------------+-----------------+-----------------+")

    return {
        "baseline_pred_mae": pred_mae_base,
        "baseline_effect_mae": effect_mae_base,
        "baseline_bias_mae": bias_mae_base,
        "baseline_consistency_mae": consistency_mae_base,
        "structured_pred_mae": pred_mae_struct,
        "structured_effect_head_mae": effect_head_mae,
        "structured_bias_head_mae": bias_head_mae,
        "structured_consistency_mae": consistency_mae,
    }


def _safe_summary(values: list[float]) -> tuple[float, float, float, int]:
    finite_vals = [v for v in values if isinstance(v, float) and torch.isfinite(torch.tensor(v))]
    if not finite_vals:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        statistics.mean(finite_vals),
        statistics.median(finite_vals),
        statistics.pstdev(finite_vals),
        len(finite_vals),
    )


def run_benchmark(num_seeds: int = 20, epochs: int = 300, start_seed: int = 0) -> None:
    """Run many seeds and summarize baseline-vs-structured prediction MAE."""
    base_vals: list[float] = []
    base_effect_vals: list[float] = []
    base_bias_vals: list[float] = []
    base_consistency_vals: list[float] = []
    struct_vals: list[float] = []
    effect_vals: list[float] = []
    bias_vals: list[float] = []
    consistency_vals: list[float] = []

    for seed in range(start_seed, start_seed + num_seeds):
        out = run_demo(seed=seed, train_epochs=epochs, verbose=False)
        base_vals.append(out["baseline_pred_mae"])
        base_effect_vals.append(out["baseline_effect_mae"])
        base_bias_vals.append(out["baseline_bias_mae"])
        base_consistency_vals.append(out["baseline_consistency_mae"])
        struct_vals.append(out["structured_pred_mae"])
        effect_vals.append(out["structured_effect_head_mae"])
        bias_vals.append(out["structured_bias_head_mae"])
        consistency_vals.append(out["structured_consistency_mae"])
        better = "structured" if out["structured_pred_mae"] < out["baseline_pred_mae"] else "baseline"
        print(
            f"seed={seed} | better={better} "
            f"baseline_pred={out['baseline_pred_mae']:.4f} "
            f"structured_pred={out['structured_pred_mae']:.4f} "
            f"baseline_eff={out['baseline_effect_mae']:.4f} "
            f"structured_eff={out['structured_effect_head_mae']:.4f} "
            f"baseline_bias={out['baseline_bias_mae']:.4f} "
            f"structured_bias={out['structured_bias_head_mae']:.4f}",
            flush=True,
        )

    wins = sum(struct_vals[i] < base_vals[i] for i in range(num_seeds))
    print("=== Prediction MAE Benchmark (baseline vs structured) ===")
    b_mean, b_med, b_std, b_n = _safe_summary(base_vals)
    be_mean, be_med, be_std, be_n = _safe_summary(base_effect_vals)
    bb_mean, bb_med, bb_std, bb_n = _safe_summary(base_bias_vals)
    bc_mean, bc_med, bc_std, bc_n = _safe_summary(base_consistency_vals)
    s_mean, s_med, s_std, s_n = _safe_summary(struct_vals)
    e_mean, e_med, e_std, e_n = _safe_summary(effect_vals)
    bi_mean, bi_med, bi_std, bi_n = _safe_summary(bias_vals)
    c_mean, c_med, c_std, c_n = _safe_summary(consistency_vals)

    print("=== End Summary Table (mean MAE over seeds) ===")
    print("+------------+-----------------+-----------------+-----------------+-----------------+")
    print("| model      | pred_mae_test   | effect_mae_test | bias_mae_test   | consistency_mae |")
    print("+------------+-----------------+-----------------+-----------------+-----------------+")
    print(
        f"| baseline   | {b_mean:>15.6f} | {be_mean:>15.6f} |"
        f" {bb_mean:>15.6f} | {bc_mean:>15.6f} |"
    )
    print(
        f"| structured | {s_mean:>15.6f} | {e_mean:>15.6f} |"
        f" {bi_mean:>15.6f} | {c_mean:>15.6f} |"
    )
    print("+------------+-----------------+-----------------+-----------------+-----------------+")
    print(f"structured better on {wins}/{num_seeds} seeds")
    print(
        f"counts used (finite): baseline(pred/effect/bias/consistency)="
        f"{b_n}/{be_n}/{bb_n}/{bc_n}, structured={s_n}/{e_n}/{bi_n}/{c_n}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    if args.num_seeds <= 1:
        run_demo(seed=args.seed, train_epochs=args.epochs, verbose=True)
    else:
        run_benchmark(num_seeds=args.num_seeds, epochs=args.epochs, start_seed=args.start_seed)


if __name__ == "__main__":
    main()
