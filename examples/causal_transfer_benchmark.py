"""Transfer benchmark: train one model across many random SEM datasets.

This benchmark is meant to mimic large tabular model evaluation:
- train a single model on many different datasets (different SEMs),
- evaluate on held-out datasets (unseen SEMs).

We compare three regimes:
1) baseline MLP predictor,
2) L2-regularized MLP predictor,
3) causal-consistency model predicting CE/CB for each observed feature.

Key challenge: each dataset may have a different number of observed variables,
so treatment count varies. We solve this via padding + masking.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from steindag.sem.dataset import SEMDatasetConfig, generate_sem_dataset
from steindag.sem.random import RandomSEMConfig

from examples.utils import (
    extract_observed_bundle,
    masked_l1,
    pad_2d,
    pad_2d_with_mask,
    permute_columns,
    print_table,
    print_table_grouped,
    summary,
)


@dataclass
class RegimeResult:
    pred_mae: float
    ce_mae: float


class Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # mask used to zero out padded features.
        x = x * mask
        return self.net(x).squeeze(-1)


class CausalConsistencyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pred_head = nn.Linear(hidden_dim, 1)
        self.ce_head = nn.Linear(hidden_dim, out_dim)
        self.cb_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = x * mask
        h = self.trunk(x)
        return (
            self.pred_head(h).squeeze(-1),
            self.ce_head(h),
            self.cb_head(h),
        )


def _evaluate_baseline(
    model: Predictor,
    x: Tensor,
    mask: Tensor,
    y: Tensor,
    ce_true: Tensor,
) -> RegimeResult:
    model.eval()
    with torch.no_grad():
        pred = model(x, mask)
        pred_mae = (pred - y).abs().mean().item()

    # CE estimate by gradient (all features)
    with torch.enable_grad():
        xg = x.detach().clone().requires_grad_(True)
        pred_g = model(xg, mask)
        grad = torch.autograd.grad(pred_g.sum(), xg, create_graph=False)[0]
        ce_pred = grad
        ce_mae = masked_l1(ce_pred, ce_true, mask).item()
    return RegimeResult(pred_mae=pred_mae, ce_mae=ce_mae)


@torch.no_grad()
def _evaluate_causal(
    model: CausalConsistencyModel,
    x: Tensor,
    mask: Tensor,
    y: Tensor,
    ce_true: Tensor,
) -> RegimeResult:
    model.eval()
    pred, ce_hat, _ = model(x, mask)
    pred_mae = (pred - y).abs().mean().item()
    ce_mae = masked_l1(ce_hat, ce_true, mask).item()
    return RegimeResult(pred_mae=pred_mae, ce_mae=ce_mae)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-datasets", type=int, default=40)
    parser.add_argument("--test-datasets", type=int, default=20)
    parser.add_argument("--samples-per-dataset", type=int, default=256)
    parser.add_argument("--min-observed", type=int, default=1)
    parser.add_argument("--num-variables", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-cb", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Build many datasets; compute max observed feature count across them.
    datasets = []
    max_t = 0
    for i in range(args.train_datasets + args.test_datasets):
        seed_i = args.seed + i
        ds = generate_sem_dataset(
            SEMDatasetConfig(
                sem_config=RandomSEMConfig(
                    num_variables=args.num_variables,
                    parent_prob=0.6,
                    nonlinear_prob=0.8,
                    sigma_range=(0.7, 1.2),
                    coef_range=(-1.0, 1.0),
                    intercept_range=(-0.5, 0.5),
                    seed=seed_i,
                ),
                num_samples=args.samples_per_dataset,
                num_queries=1,
                min_observed=args.min_observed,
                seed=seed_i,
            )
        )
        feature_names, _, _, _, _ = extract_observed_bundle(ds, strategy="first")
        max_t = max(max_t, len(feature_names))
        datasets.append(ds)

    train_datasets = datasets[: args.train_datasets]
    test_datasets = datasets[args.train_datasets :]

    # Build flattened training data with padding.
    x_train_all: list[Tensor] = []
    m_train_all: list[Tensor] = []
    y_train_all: list[Tensor] = []
    ce_train_all: list[Tensor] = []
    cb_train_all: list[Tensor] = []
    for ds in train_datasets:
        feature_names, _, y_all, ce_all, cb_all = extract_observed_bundle(
            ds, strategy="first"
        )
        x_all = torch.stack([ds.data[name] for name in feature_names], dim=1)
        x_pad, mask = pad_2d_with_mask(x_all, max_t)
        ce_pad = pad_2d(ce_all, max_t)
        cb_pad = pad_2d(cb_all, max_t)

        # Permutation augmentation: destroy positional shortcuts by permuting
        # feature order (and aligned causal targets) per dataset.
        g = torch.Generator(device=x_pad.device)
        g.manual_seed(args.seed + 10_000 + len(x_train_all))
        x_pad, mask, (ce_pad, cb_pad) = permute_columns(
            x_pad, mask, ce_pad, cb_pad, generator=g
        )

        # Normalize per-dataset feature scale to mimic tabular preprocessing.
        mean = x_pad.mean(dim=0, keepdim=True)
        std = x_pad.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_pad = (x_pad - mean) / std

        x_train_all.append(x_pad)
        m_train_all.append(mask)
        y_train_all.append(y_all)
        ce_train_all.append(ce_pad)
        cb_train_all.append(cb_pad)

    X = torch.cat(x_train_all, dim=0)
    M = torch.cat(m_train_all, dim=0)
    Y = torch.cat(y_train_all, dim=0)
    CE = torch.cat(ce_train_all, dim=0)
    CB = torch.cat(cb_train_all, dim=0)

    # Models.
    baseline = Predictor(in_dim=max_t)
    l2 = Predictor(in_dim=max_t)
    causal = CausalConsistencyModel(in_dim=max_t, out_dim=max_t)

    opt_base = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    opt_l2 = torch.optim.Adam(
        l2.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    opt_causal = torch.optim.Adam(causal.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    loader = DataLoader(
        list(zip(X, M, Y, CE, CB)), batch_size=args.batch_size, shuffle=True
    )

    for _ in range(args.epochs):
        baseline.train()
        l2.train()
        causal.train()
        for xb, mb, yb, ceb, cbb in loader:
            # Baseline
            opt_base.zero_grad()
            pred_b = baseline(xb, mb)
            loss_b = mse(pred_b, yb)
            loss_b.backward()
            opt_base.step()

            # L2
            opt_l2.zero_grad()
            pred_l2 = l2(xb, mb)
            loss_l2 = mse(pred_l2, yb)
            loss_l2.backward()
            opt_l2.step()

            # Causal
            xc = xb.detach().clone().requires_grad_(True)
            opt_causal.zero_grad()
            pred_c, ce_hat, cb_hat = causal(xc, mb)
            loss_pred = mse(pred_c, yb)
            loss_ce = masked_l1(ce_hat, ceb, mb)
            loss_cb = masked_l1(cb_hat, cbb, mb)
            grad_pred = torch.autograd.grad(pred_c.sum(), xc, create_graph=True)[0]
            loss_cons = masked_l1(grad_pred, ce_hat + cb_hat, mb)
            loss = (
                loss_pred
                + args.lambda_ce * loss_ce
                + args.lambda_cb * loss_cb
                + args.lambda_consistency * loss_cons
            )
            loss.backward()
            opt_causal.step()

    # Evaluation per dataset.
    per_ds_rows: list[list[str]] = []
    per_regime_pred: dict[str, list[float]] = {"baseline": [], "l2": [], "causal": []}
    per_regime_ce: dict[str, list[float]] = {"baseline": [], "l2": [], "causal": []}

    for j, ds in enumerate(test_datasets):
        feature_names, _, y_all, ce_all, _cb_all = extract_observed_bundle(
            ds, strategy="first"
        )
        x_all = torch.stack([ds.data[name] for name in feature_names], dim=1)
        x_pad, mask = pad_2d_with_mask(x_all, max_t)
        ce_pad = pad_2d(ce_all, max_t)

        # Permute at test time as well (consistent with training invariance goal).
        g = torch.Generator(device=x_pad.device)
        g.manual_seed(args.seed + 20_000 + j)
        x_pad, mask, (ce_pad,) = permute_columns(x_pad, mask, ce_pad, generator=g)

        mean = x_pad.mean(dim=0, keepdim=True)
        std = x_pad.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_pad = (x_pad - mean) / std

        r_base = _evaluate_baseline(baseline, x_pad, mask, y_all, ce_pad)
        r_l2 = _evaluate_baseline(l2, x_pad, mask, y_all, ce_pad)
        r_causal = _evaluate_causal(causal, x_pad, mask, y_all, ce_pad)

        for name, res in [
            ("baseline", r_base),
            ("l2", r_l2),
            ("causal", r_causal),
        ]:
            per_regime_pred[name].append(res.pred_mae)
            per_regime_ce[name].append(res.ce_mae)
            per_ds_rows.append(
                [
                    f"{j:02d}",
                    f"{len(feature_names)}",
                    name,
                    f"{res.pred_mae:.6f}",
                    f"{res.ce_mae:.6f}",
                ]
            )

    print_table_grouped(
        "Test dataset results",
        [
            "test_ds",
            "T",
            "regime",
            "prediction_mae",
            "causal_effect_mae_avg_over_treatments",
        ],
        per_ds_rows,
        group_col=0,
    )

    summary_rows: list[list[str]] = []
    for regime in ["baseline", "l2", "causal"]:
        p = summary(per_regime_pred[regime])
        c = summary(per_regime_ce[regime])
        summary_rows.append(
            [
                regime,
                f"{p['mean']:.6f}",
                f"{p['std']:.6f}",
                f"{c['mean']:.6f}",
                f"{c['std']:.6f}",
            ]
        )

    print_table(
        "Transfer summary (across test datasets)",
        [
            "regime",
            "pred_mean",
            "pred_std",
            "ce_mean",
            "ce_std",
        ],
        summary_rows,
    )


if __name__ == "__main__":
    main()
