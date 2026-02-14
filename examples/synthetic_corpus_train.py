"""Train transfer models on a stored synthetic corpus and save checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from steindag.sem.dataset import load_sem_dataset

from examples.transfer_models import CausalConsistencyModel, ModelCheckpoint, Predictor
from examples.utils import (
    extract_observed_bundle,
    masked_l1,
    pad_2d,
    pad_2d_with_mask,
    permute_columns,
)


def _math_sdpa_context():
    """Prefer math SDPA backend for higher-order grads on CPU.

    Some optimized SDPA kernels (including CPU flash variants in recent
    PyTorch) do not implement the backward-of-backward path required when
    `create_graph=True` is used to enforce gradient consistency.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return nullcontext()


def _masked_mean_std(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compute per-feature mean/std for a padded matrix using a (0/1) mask."""
    # x, mask: (N,D)
    denom = mask.sum(dim=0, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=0, keepdim=True) / denom
    var = ((x - mean) ** 2 * mask).sum(dim=0, keepdim=True) / denom
    std = var.sqrt().clamp_min(1e-6)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=".data/synth_corpus",
        help="Directory created by synthetic_corpus_generate.py (must contain manifest.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".data/synth_models",
        help="Where to write baseline.pt / l2.pt / causal.pt + meta.json",
    )
    parser.add_argument("--max-datasets", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--arch", type=str, default="ft_transformer_v1")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--adapter-dim",
        type=int,
        default=32,
        help="Adapter bottleneck dim for ft_transformer_v1. Use 0 to disable adapters.",
    )
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-cb", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=1.0)

    # Optional meta-learning (Reptile) to learn an initialization that adapts
    # quickly to new datasets (few-shot transfer).
    parser.add_argument(
        "--meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Train the causal model with Reptile meta-learning instead of pooled ERM. "
            "Default: --meta (enabled), because this project focuses on few-shot transfer."
        ),
    )
    parser.add_argument("--meta-epochs", type=int, default=3)
    parser.add_argument("--meta-lr", type=float, default=0.05)
    parser.add_argument("--inner-steps", type=int, default=10)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--support-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Robustness on CPU: disable Flash/MemEff SDPA kernels because second-order
    # gradients (used by the consistency term via create_graph=True) are not
    # implemented for some CPU flash-attention backends.
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    torch.manual_seed(args.seed)

    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    entries = manifest["datasets"][: args.max_datasets]

    datasets = [load_sem_dataset(corpus_dir / e["path"]) for e in entries]

    # Determine max observed feature count.
    max_t = 0
    for ds in datasets:
        feature_names, _, _, _, _ = extract_observed_bundle(ds, strategy="max_treatments")
        max_t = max(max_t, len(feature_names))

    # Flatten training data (pooled ERM) and also keep per-task tensors (meta).
    x_all: list[Tensor] = []
    m_all: list[Tensor] = []
    y_all: list[Tensor] = []
    ce_all: list[Tensor] = []
    cb_all: list[Tensor] = []
    tasks: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = []
    for i, ds in enumerate(datasets):
        feature_names, _, y, ce, cb = extract_observed_bundle(ds, strategy="max_treatments")
        x = torch.stack([ds.data[name] for name in feature_names], dim=1)
        x_pad, mask = pad_2d_with_mask(x, max_t)
        ce_pad = pad_2d(ce, max_t)
        cb_pad = pad_2d(cb, max_t)

        # Permute per dataset.
        g = torch.Generator(device=x_pad.device)
        g.manual_seed(args.seed + 10_000 + i)
        x_pad, mask, (ce_pad, cb_pad) = permute_columns(
            x_pad, mask, ce_pad, cb_pad, generator=g
        )

        # Per-dataset normalization (masked so padding doesn't skew stats).
        mean, std = _masked_mean_std(x_pad, mask)
        x_pad = (x_pad - mean) / std

        # Important: CE/CB are derivatives w.r.t the *original* x.
        # After normalization x_norm=(x-mean)/std, gradients transform as:
        #   d y / d x_norm = (d y / d x) * std
        # so we must scale targets by std.
        ce_pad = ce_pad * std
        cb_pad = cb_pad * std

        # Per-task y standardization + consistent scaling of (CE,CB)
        y_mean_task = y.mean()
        y_std_task = y.std().clamp_min(1e-6)
        y_task = (y - y_mean_task) / y_std_task
        ce_pad_task = ce_pad / y_std_task
        cb_pad_task = cb_pad / y_std_task

        tasks.append((x_pad, mask, y_task, ce_pad_task, cb_pad_task))

        x_all.append(x_pad)
        m_all.append(mask)
        y_all.append(y)
        ce_all.append(ce_pad)
        cb_all.append(cb_pad)

    X = torch.cat(x_all, dim=0)
    M = torch.cat(m_all, dim=0)
    Y = torch.cat(y_all, dim=0)
    CE = torch.cat(ce_all, dim=0)
    CB = torch.cat(cb_all, dim=0)

    # Standardize target (pooled ERM mode) to make synthetic pretraining transferable and stable.
    # If y_std is large on real datasets (e.g. airfoil), this avoids massive
    # scale mismatch when fine-tuning.
    y_mean = Y.mean()
    y_std = Y.std().clamp_min(1e-6)
    Y = (Y - y_mean) / y_std
    # CE and CB are derivatives of y; scaling y by 1/y_std scales (CE,CB) too.
    CE = CE / y_std
    CB = CB / y_std

    model_kwargs = {
        "nhead": int(args.nhead),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "adapter_dim": int(args.adapter_dim),
    }

    if args.arch == "ft_transformer_v1":
        baseline = Predictor(in_dim=max_t, hidden_dim=args.hidden_dim, **model_kwargs)
        l2 = Predictor(in_dim=max_t, hidden_dim=args.hidden_dim, **model_kwargs)
        causal = CausalConsistencyModel(
            in_dim=max_t, out_dim=max_t, hidden_dim=args.hidden_dim, **model_kwargs
        )
    elif args.arch == "deepsets_v1":
        # DeepSets variants are still accessible via build_model_from_checkpoint,
        # but for training we use the default (Transformer) unless requested.
        from examples.transfer_models import DeepSetsCausalConsistencyModel, DeepSetsPredictor

        baseline = DeepSetsPredictor(in_dim=max_t, hidden_dim=args.hidden_dim)
        l2 = DeepSetsPredictor(in_dim=max_t, hidden_dim=args.hidden_dim)
        causal = DeepSetsCausalConsistencyModel(
            in_dim=max_t, out_dim=max_t, hidden_dim=args.hidden_dim
        )
        model_kwargs = {}
    else:
        raise ValueError("--arch must be one of: ft_transformer_v1, deepsets_v1")

    opt_base = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    opt_l2 = torch.optim.Adam(l2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_causal = torch.optim.Adam(causal.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    if not args.meta:
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
                loss_b = mse(baseline(xb, mb), yb)
                loss_b.backward()
                opt_base.step()

                # L2
                opt_l2.zero_grad()
                loss_l2 = mse(l2(xb, mb), yb)
                loss_l2.backward()
                opt_l2.step()

                # Causal
                xc = xb.detach().clone().requires_grad_(True)
                opt_causal.zero_grad()
                with _math_sdpa_context():
                    pred_c, ce_hat, cb_hat = causal(xc, mb)
                loss_pred = mse(pred_c, yb)
                loss_ce = masked_l1(ce_hat, ceb, mb)
                loss_cb = masked_l1(cb_hat, cbb, mb)
                with _math_sdpa_context():
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
    else:
        if not (0.0 < args.support_frac <= 1.0):
            raise ValueError("--support-frac must be in (0,1]")

        # Train baseline/l2 with pooled ERM anyway (they are just baselines).
        loader = DataLoader(list(zip(X, M, Y, CE, CB)), batch_size=args.batch_size, shuffle=True)
        for _ in range(args.epochs):
            baseline.train(); l2.train()
            for xb, mb, yb, _ceb, _cbb in loader:
                opt_base.zero_grad(); mse(baseline(xb, mb), yb).backward(); opt_base.step()
                opt_l2.zero_grad(); mse(l2(xb, mb), yb).backward(); opt_l2.step()

        # Reptile meta-learning for the causal model.
        causal.train()
        for _ in range(args.meta_epochs):
            # Iterate tasks in random order.
            order = torch.randperm(len(tasks)).tolist()
            for ti in order:
                xb, mb, yb, ceb, cbb = tasks[ti]

                # Support set
                n = xb.shape[0]
                n_sup = max(8, int(round(args.support_frac * n)))
                sup_x = xb[:n_sup]
                sup_m = mb[:n_sup]
                sup_y = yb[:n_sup]
                sup_ce = ceb[:n_sup]
                sup_cb = cbb[:n_sup]

                fast = copy.deepcopy(causal)
                inner_opt = torch.optim.Adam(fast.parameters(), lr=args.inner_lr)

                for _ in range(args.inner_steps):
                    inner_opt.zero_grad()
                    xg = sup_x.detach().clone().requires_grad_(True)
                    with _math_sdpa_context():
                        pred, ce_hat, cb_hat = fast(xg, sup_m)
                    loss_pred = mse(pred, sup_y)
                    loss_ce = masked_l1(ce_hat, sup_ce, sup_m)
                    loss_cb = masked_l1(cb_hat, sup_cb, sup_m)
                    with _math_sdpa_context():
                        grad_pred = torch.autograd.grad(pred.sum(), xg, create_graph=True)[0]
                    loss_cons = masked_l1(grad_pred, ce_hat + cb_hat, sup_m)
                    loss = (
                        loss_pred
                        + args.lambda_ce * loss_ce
                        + args.lambda_cb * loss_cb
                        + args.lambda_consistency * loss_cons
                    )
                    loss.backward()
                    inner_opt.step()

                # Reptile outer update: move initialization towards fast weights.
                with torch.no_grad():
                    for p, p_fast in zip(causal.parameters(), fast.parameters()):
                        p.add_(args.meta_lr * (p_fast - p))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ModelCheckpoint(
        model_type="baseline",
        arch=str(args.arch),
        in_dim=max_t,
        out_dim=None,
        hidden_dim=args.hidden_dim,
        model_kwargs=model_kwargs,
        state_dict=baseline.state_dict(),
    ).save(out_dir / "baseline.pt")
    ModelCheckpoint(
        model_type="l2",
        arch=str(args.arch),
        in_dim=max_t,
        out_dim=None,
        hidden_dim=args.hidden_dim,
        model_kwargs=model_kwargs,
        state_dict=l2.state_dict(),
    ).save(out_dir / "l2.pt")
    ModelCheckpoint(
        model_type="causal",
        arch=str(args.arch),
        in_dim=max_t,
        out_dim=max_t,
        hidden_dim=args.hidden_dim,
        model_kwargs=model_kwargs,
        state_dict=causal.state_dict(),
    ).save(out_dir / "causal.pt")

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "format": "steindag-synth-models-v2",
                "trained_on": {
                    "corpus_dir": str(corpus_dir),
                    "num_datasets": len(datasets),
                    "max_t": max_t,
                },
                "target_standardization": {
                    "y_mean": float(y_mean.item()),
                    "y_std": float(y_std.item()),
                },
                "args": vars(args),
                "meta_learning": {
                    "enabled": bool(args.meta),
                    "algorithm": "reptile" if args.meta else None,
                },
            },
            indent=2,
        )
    )

    print(f"Saved model checkpoints to: {out_dir}")


if __name__ == "__main__":
    main()
