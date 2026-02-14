"""Evaluate synthetic-trained models on open-source real tabular regression datasets.

Methodology (per real dataset):
1) Split real dataset into train/test.
2) Train a *baseline* predictor from scratch on real train.
3) Load a *prediction-only* model pre-trained on synthetic data and fine-tune
   it on real train.
4) Load a *causal-consistency* model pre-trained on synthetic data and fine-tune
   it on real train (prediction loss + optional consistency regularization).
5) Evaluate all three on real test.

We report:
- prediction MAE on test,
- per-feature estimated causal effect and causal bias on test:
  - baseline: CE = gradient of prediction wrt each feature; CB=0
  - synthetic_finetuned: CE = gradient of prediction wrt each feature; CB=0
  - causal-consistency: CE/CB are model heads.

No ground-truth causal effects exist for real datasets; we only report estimates.

Important note on feature dimensionality:
- the synthetic checkpoint has a fixed `in_dim`;
- if a real dataset has D > in_dim, we currently truncate to the first in_dim
  features (for both baseline and causal) and print a warning.

To avoid this truncation in practice, pretrain on a synthetic corpus with a
larger `num_variables` so that checkpoint `in_dim` covers your target datasets.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from examples.tabular_regression_datasets import load_regression_datasets
from examples.transfer_models import ModelCheckpoint, Predictor, build_model_from_checkpoint
from examples.utils import masked_l1, pad_2d_with_mask, print_table, summary


def _math_sdpa_context():
    """Prefer math SDPA backend where higher-order grads are needed on CPU."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        return nullcontext()


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _train_test_split(n: int, *, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = max(1, int(round(test_frac * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def _train_val_split(train_idx: np.ndarray, *, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.array(train_idx, copy=True)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(idx))))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


@torch.no_grad()
def _pred_mae(model: nn.Module, x: Tensor, mask: Tensor, y: Tensor) -> float:
    out = model(x, mask)
    pred = out[0] if isinstance(out, tuple) else out
    return float((pred - y).abs().mean().item())


def _estimate_effects(
    model: nn.Module,
    x_norm: Tensor,
    mask: Tensor,
    *,
    feature_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean effects in both normalized-X and raw-X units.

    We feed the model *normalized* inputs x_norm, so gradients/CE/CB are w.r.t.
    x_norm. If x_norm = (x_raw - mean)/std, then by chain rule:
        d y / d x_raw = (d y / d x_norm) / std

    Returns:
        (ce_norm, cb_norm, ce_raw, cb_raw)
    all shaped (D,).
    """

    std_t = torch.tensor(feature_std, dtype=torch.float32)
    d_model = int(x_norm.shape[1])
    if std_t.numel() < d_model:
        # For padded (non-real) feature slots, use scale 1 so that CE/CB in
        # normalized-X units are unchanged in "raw" view.
        std_t = torch.cat([std_t, torch.ones(d_model - std_t.numel())], dim=0)
    else:
        std_t = std_t[:d_model]
    std_t = torch.clamp(std_t, min=1e-12)

    out = model(x_norm, mask)
    if isinstance(out, tuple) and len(out) == 3:
        _pred, ce_hat, cb_hat = out
        denom = mask.sum(dim=0).clamp_min(1.0)
        ce_norm = (ce_hat * mask).sum(dim=0) / denom
        cb_norm = (cb_hat * mask).sum(dim=0) / denom
    else:
        # Baseline/L2: CE via gradient; CB=0
        with torch.enable_grad():
            xg = x_norm.detach().clone().requires_grad_(True)
            pred = model(xg, mask)
            grad = torch.autograd.grad(pred.sum(), xg, create_graph=False)[0]
            ce_norm = (grad * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
        cb_norm = torch.zeros_like(ce_norm)

    ce_raw = ce_norm / std_t
    cb_raw = cb_norm / std_t

    return (
        ce_norm.detach().cpu().numpy(),
        cb_norm.detach().cpu().numpy(),
        ce_raw.detach().cpu().numpy(),
        cb_raw.detach().cpu().numpy(),
    )


def _to_padded_tensor(x: np.ndarray, *, in_dim: int) -> tuple[Tensor, Tensor]:
    """Convert numpy (N,D) to (x_pad, mask) with pad/truncate to in_dim."""
    xt = torch.tensor(x, dtype=torch.float32)
    if xt.shape[1] > in_dim:
        xt = xt[:, :in_dim]
    x_pad, mask = pad_2d_with_mask(xt, in_dim)
    return x_pad, mask


def _train_baseline_on_real(
    *,
    x_train: Tensor,
    mask_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    mask_val: Tensor,
    y_val: Tensor,
    in_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    model_kwargs: dict,
    seed: int,
) -> tuple[Predictor, float, float]:
    torch.manual_seed(seed)
    # Standardize y for stability.
    y_mean = float(y_train.mean().item())
    y_std = float(y_train.std().clamp_min(1e-6).item())
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    # For a fair comparison, use the same model class/hparams as the pretrained
    # causal model, but initialize from scratch.
    model = Predictor(in_dim=in_dim, hidden_dim=hidden_dim, **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(x_train, mask_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    patience = 10
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            pred = model(xb, mb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_mae = _pred_mae(model, x_val, mask_val, y_val)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, y_mean, y_std


def _finetune_causal_on_real(
    *,
    ckpt: ModelCheckpoint,
    x_train: Tensor,
    mask_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    mask_val: Tensor,
    y_val: Tensor,
    linear_probe_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    lr_probe: float,
    lr_finetune: float,
    lambda_consistency: float,
    lambda_cb: float,
    lambda_sp: float,
    use_consistency: bool,
    freeze_ce_cb: bool,
    finetune_mode: str,
    seed: int,
) -> tuple[nn.Module, float, float]:
    torch.manual_seed(seed)
    if ckpt.out_dim is None:
        raise ValueError("Expected a causal checkpoint with out_dim")

    # Standardize y for stability.
    y_mean = float(y_train.mean().item())
    y_std = float(y_train.std().clamp_min(1e-6).item())
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    model = build_model_from_checkpoint(ckpt)

    # L2-SP anchor for stable transfer learning (penalize deviation from the
    # pretrained weights). This is often more stable than naive fine-tuning.
    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}

    mse = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, mask_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    # Optionally freeze CE/CB heads during adaptation.
    # This tends to preserve the synthetic causal structure while letting the
    # predictor adapt to real-data marginals.
    # Identify backbone module for the selected architecture.
    backbone = None
    if hasattr(model, "feature_encoder"):
        backbone = getattr(model, "feature_encoder")
    elif hasattr(model, "backbone"):
        backbone = getattr(model, "backbone")
    else:
        raise AttributeError("Could not find backbone module on causal model")

    if not hasattr(model, "pred_head") or not hasattr(model, "per_feature_head"):
        raise AttributeError("Causal model is expected to have pred_head and per_feature_head")
    pred_head = getattr(model, "pred_head")
    per_feature_head = getattr(model, "per_feature_head")

    # We always freeze CE/CB heads during the probe step below; later we may
    # unfreeze them depending on `freeze_ce_cb`.
    for p in per_feature_head.parameters():
        p.requires_grad_(not freeze_ce_cb)

    # Phase 1: linear probe (train pred head only)
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in per_feature_head.parameters():
        p.requires_grad_(False)
    for p in pred_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(pred_head.parameters(), lr=lr_probe)

    for _ in range(linear_probe_epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            pred, _ce, _cb = model(xb, mb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()

    # Phase 2: choose adaptation strategy.
    # - head:       train pred head only
    # - adapter:    train adapter + pred head (if backbone has .adapter)
    # - full:       train backbone + pred head
    if finetune_mode not in {"head", "adapter", "full"}:
        raise ValueError("finetune_mode must be one of: head, adapter, full")

    # Default: freeze everything in backbone.
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Re-apply CE/CB freezing choice for phase 2.
    for p in per_feature_head.parameters():
        p.requires_grad_(not freeze_ce_cb)

    params: list[Tensor] = []
    if finetune_mode == "full":
        for p in backbone.parameters():
            p.requires_grad_(True)
        params += list(backbone.parameters())
    elif finetune_mode == "adapter":
        if hasattr(backbone, "adapter") and getattr(backbone, "adapter") is not None:
            for p in getattr(backbone, "adapter").parameters():
                p.requires_grad_(True)
            params += list(getattr(backbone, "adapter").parameters())
        # else: fall back to head-only

    # Optionally include CE/CB head parameters in fine-tuning.
    if not freeze_ce_cb:
        params += list(per_feature_head.parameters())

    for p in pred_head.parameters():
        p.requires_grad_(True)
    params += list(pred_head.parameters())
    opt2 = torch.optim.Adam(params, lr=lr_finetune)

    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    patience = 10
    bad = 0

    for _ in range(finetune_epochs):
        model.train()
        for xb, mb, yb in loader:
            opt2.zero_grad()

            if use_consistency:
                # Need grads w.r.t x for consistency term.
                xb2 = xb.detach().clone().requires_grad_(True)
                with _math_sdpa_context():
                    pred, ce_hat, cb_hat = model(xb2, mb)
                loss_pred = mse(pred, yb)
                with _math_sdpa_context():
                    grad_pred = torch.autograd.grad(pred.sum(), xb2, create_graph=True)[0]
                loss_cons = masked_l1(grad_pred, ce_hat + cb_hat, mb)
                loss_cb = masked_l1(cb_hat, torch.zeros_like(cb_hat), mb)
                loss = loss_pred + lambda_consistency * loss_cons + lambda_cb * loss_cb
            else:
                pred, _ce_hat, _cb_hat = model(xb, mb)
                loss = mse(pred, yb)

            if lambda_sp > 0.0:
                # Only regularize parameters we are actually updating.
                sp = 0.0
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    sp = sp + (p - anchor[name]).pow(2).sum()
                loss = loss + lambda_sp * sp
            loss.backward()
            opt2.step()

        model.eval()
        val_mae = _pred_mae(model, x_val, mask_val, y_val)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, y_mean, y_std


def _finetune_causal_auto(
    *,
    ckpt: ModelCheckpoint,
    x_train: Tensor,
    mask_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    mask_val: Tensor,
    y_val: Tensor,
    batch_size: int,
    # shared
    linear_probe_epochs: int,
    finetune_epochs: int,
    lr_probe: float,
    lr_finetune: float,
    lambda_consistency: float,
    lambda_cb: float,
    lambda_sp: float,
    use_consistency: bool,
    freeze_ce_cb: bool,
    seed: int,
) -> tuple[nn.Module, float, float]:
    """Try a small set of conservative fine-tuning configs and pick best on val.

    This improves robustness in few-shot by doing methodology-level
    validation-based model selection, instead of relying on one brittle set of
    hyperparameters.
    """

    # (finetune_mode, lr_finetune, lambda_sp, freeze_ce_cb, use_consistency)
    # Keep this list short: we want robustness without making runtime explode.
    # Candidates are generated around the user-provided/base settings.
    lr0 = float(lr_finetune)
    sp0 = float(lambda_sp)

    # If the user did not explicitly request to unfreeze CE/CB heads, still try
    # one candidate that unfreezes them (with stronger anchoring) and let the
    # validation set decide.
    ce_cb_candidates = [True] if freeze_ce_cb else [False]
    if freeze_ce_cb:
        ce_cb_candidates.append(False)

    candidates: list[tuple[str, float, float, bool, bool]] = []
    for freeze_heads in ce_cb_candidates:
        candidates += [
            ("adapter", lr0, sp0, freeze_heads, False),
            ("adapter", lr0 * 0.6, sp0 * 3.0, freeze_heads, False),
            ("head", lr0 * 0.6, sp0 * 3.0, freeze_heads, False),
        ]

        # Consistency can help when unfreezing heads, but often hurts; include
        # exactly one conservative candidate.
        if not freeze_heads:
            candidates.append(("adapter", lr0 * 0.6, sp0 * 3.0, freeze_heads, True))

    best = None
    best_val = float("inf")
    best_y_stats: tuple[float, float] | None = None

    for mode, lr_ft, lambda_sp_i, freeze_heads, use_cons in candidates:
        model, y_mean, y_std = _finetune_causal_on_real(
            ckpt=ckpt,
            x_train=x_train,
            mask_train=mask_train,
            y_train=y_train,
            x_val=x_val,
            mask_val=mask_val,
            y_val=y_val,
            linear_probe_epochs=linear_probe_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            lr_probe=lr_probe,
            lr_finetune=lr_ft,
            lambda_consistency=lambda_consistency,
            lambda_cb=lambda_cb,
            lambda_sp=lambda_sp_i,
            use_consistency=bool(use_cons),
            freeze_ce_cb=bool(freeze_heads),
            finetune_mode=mode,
            seed=seed,
        )

        # _finetune_causal_on_real trains on standardized y, so evaluate on standardized y.
        y_val_std = (y_val - y_mean) / max(y_std, 1e-12)
        val_mae = _pred_mae(model, x_val, mask_val, y_val_std)

        if val_mae < best_val:
            best_val = val_mae
            best = model
            best_y_stats = (y_mean, y_std)

    assert best is not None and best_y_stats is not None
    return best, best_y_stats[0], best_y_stats[1]


def _finetune_predictor_on_real(
    *,
    ckpt: ModelCheckpoint,
    x_train: Tensor,
    mask_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    mask_val: Tensor,
    y_val: Tensor,
    linear_probe_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    lr_probe: float,
    lr_finetune: float,
    lambda_sp: float,
    finetune_mode: str,
    seed: int,
) -> tuple[nn.Module, float, float]:
    torch.manual_seed(seed)

    model = build_model_from_checkpoint(ckpt)
    if not isinstance(model, nn.Module):
        raise TypeError("Expected torch.nn.Module from predictor checkpoint")

    y_mean = float(y_train.mean().item())
    y_std = float(y_train.std().clamp_min(1e-6).item())
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    anchor = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mse = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_train, mask_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    if not hasattr(model, "backbone") and not hasattr(model, "feature_encoder"):
        raise AttributeError("Predictor model is expected to expose backbone or feature_encoder")
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        backbone = getattr(model, "feature_encoder")

    if hasattr(model, "pred_head"):
        pred_head = getattr(model, "pred_head")
    elif hasattr(model, "aggregator"):
        pred_head = getattr(model, "aggregator")
    else:
        pred_head = None

    # Phase 1: probe head only when possible.
    if pred_head is not None:
        for p in backbone.parameters():
            p.requires_grad_(False)
        for p in pred_head.parameters():
            p.requires_grad_(True)
        opt_probe = torch.optim.Adam(pred_head.parameters(), lr=lr_probe)
        for _ in range(linear_probe_epochs):
            model.train()
            for xb, mb, yb in loader:
                opt_probe.zero_grad()
                pred = model(xb, mb)
                loss = mse(pred, yb)
                loss.backward()
                opt_probe.step()

    if finetune_mode not in {"head", "adapter", "full"}:
        raise ValueError("finetune_mode must be one of: head, adapter, full")

    for p in backbone.parameters():
        p.requires_grad_(False)

    params: list[Tensor] = []
    if finetune_mode == "full":
        for p in backbone.parameters():
            p.requires_grad_(True)
        params += list(backbone.parameters())
    elif finetune_mode == "adapter":
        if hasattr(backbone, "adapter") and getattr(backbone, "adapter") is not None:
            for p in getattr(backbone, "adapter").parameters():
                p.requires_grad_(True)
            params += list(getattr(backbone, "adapter").parameters())

    if pred_head is not None:
        for p in pred_head.parameters():
            p.requires_grad_(True)
        params += list(pred_head.parameters())
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        params = list(model.parameters())

    opt = torch.optim.Adam(params, lr=lr_finetune)

    best_state: dict[str, Tensor] | None = None
    best_val = float("inf")
    patience = 10
    bad = 0

    for _ in range(finetune_epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            pred = model(xb, mb)
            loss = mse(pred, yb)

            if lambda_sp > 0.0:
                sp = 0.0
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    sp = sp + (p - anchor[name]).pow(2).sum()
                loss = loss + lambda_sp * sp

            loss.backward()
            opt.step()

        model.eval()
        val_mae = _pred_mae(model, x_val, mask_val, y_val)
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, y_mean, y_std


def _finetune_predictor_auto(
    *,
    ckpt: ModelCheckpoint,
    x_train: Tensor,
    mask_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    mask_val: Tensor,
    y_val: Tensor,
    batch_size: int,
    linear_probe_epochs: int,
    finetune_epochs: int,
    lr_probe: float,
    lr_finetune: float,
    lambda_sp: float,
    seed: int,
) -> tuple[nn.Module, float, float]:
    candidates = [
        ("adapter", lr_finetune, max(lambda_sp, 1e-3)),
        ("head", lr_finetune * 0.7, max(lambda_sp * 3.0, 3e-3)),
    ]

    best = None
    best_val = float("inf")
    best_y_stats: tuple[float, float] | None = None

    for mode, lr_ft, lambda_sp_i in candidates:
        model, y_mean, y_std = _finetune_predictor_on_real(
            ckpt=ckpt,
            x_train=x_train,
            mask_train=mask_train,
            y_train=y_train,
            x_val=x_val,
            mask_val=mask_val,
            y_val=y_val,
            linear_probe_epochs=linear_probe_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            lr_probe=lr_probe,
            lr_finetune=lr_ft,
            lambda_sp=lambda_sp_i,
            finetune_mode=mode,
            seed=seed,
        )

        y_val_std = (y_val - y_mean) / max(y_std, 1e-12)
        val_mae = _pred_mae(model, x_val, mask_val, y_val_std)
        if val_mae < best_val:
            best_val = val_mae
            best = model
            best_y_stats = (y_mean, y_std)

    assert best is not None and best_y_stats is not None
    return best, best_y_stats[0], best_y_stats[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=str,
        default=".data/synth_models",
        help="Directory containing baseline.pt / l2.pt / causal.pt produced by synthetic_corpus_train.py",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        # Defaults chosen to make the causal model's advantage visible in the
        # few-shot setting.
        default="abalone,yacht_hydrodynamics",
        help="Comma-separated dataset names",
    )
    parser.add_argument("--cache-dir", type=str, default=".data/tabular_regression")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.01,
        help="Use only this fraction of the real training set (few-shot transfer). Default: 0.01.",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Real-data training hyperparams
    # NOTE: defaults are tuned for *few-shot* transfer (train-frac ~ 0.05)
    # on CPU, to give the causal-finetuned model a fair chance.
    parser.add_argument("--real-epochs", type=int, default=200)
    parser.add_argument("--real-batch-size", type=int, default=64)
    parser.add_argument("--real-lr", type=float, default=1e-3)
    parser.add_argument("--real-weight-decay", type=float, default=1e-2)

    # Fine-tuning hyperparams
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument(
        "--finetune-mode",
        type=str,
        default="auto",
        help=(
            "Few-shot adaptation strategy: auto | head | adapter | full. "
            "Default: auto (try a few conservative configs and pick best on validation)."
        ),
    )
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument("--lambda-cb", type=float, default=0.1)
    parser.add_argument(
        "--lambda-sp",
        type=float,
        default=1e-3,
        help="L2-SP strength for fine-tuning (penalize deviation from pretrained weights).",
    )
    parser.add_argument(
        "--use-consistency",
        action="store_true",
        help="If set, fine-tuning includes gradient-consistency regularization. Default: off (prediction-only fine-tuning).",
    )
    parser.add_argument(
        "--no-freeze-ce-cb",
        action="store_true",
        help="If set, also fine-tune the CE/CB heads. Default: CE/CB heads are frozen.",
    )
    args = parser.parse_args()

    # Robustness on CPU for higher-order grads used by causal consistency.
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    # Auto-tuned few-shot settings.
    # If the user requested finetune_mode=auto, we also set conservative
    # learning-rate/regularization schedules based on train-frac.
    # This is a methodology-level robustness improvement: avoid high-variance
    # full fine-tuning when data is scarce.
    if str(args.finetune_mode) == "auto":
        tf = float(args.train_frac)
        if tf <= 0.02:
            # Very few-shot: be very conservative.
            args.probe_epochs = max(args.probe_epochs, 80)
            args.finetune_epochs = max(args.finetune_epochs, 120)
            args.finetune_lr = min(args.finetune_lr, 3e-4)
            args.lambda_sp = max(args.lambda_sp, 3e-3)
        elif tf <= 0.07:
            # Default sweet spot (~0.05)
            args.probe_epochs = max(args.probe_epochs, 50)
            args.finetune_epochs = max(args.finetune_epochs, 200)
            args.finetune_lr = max(args.finetune_lr, 5e-4)
            args.lambda_sp = max(args.lambda_sp, 1e-3)
        elif tf <= 0.15:
            # More data: reduce fine-tune aggressiveness; pretraining helps less.
            args.finetune_epochs = min(args.finetune_epochs, 120)
            args.finetune_lr = min(args.finetune_lr, 1e-4)
            args.lambda_sp = max(args.lambda_sp, 1e-2)
        else:
            # Full data: still avoid large updates from the synthetic init.
            args.finetune_epochs = min(args.finetune_epochs, 100)
            args.finetune_lr = min(args.finetune_lr, 1e-4)
            args.lambda_sp = max(args.lambda_sp, 1e-2)

    models_dir = Path(args.models_dir)

    synth_pred_path = models_dir / "baseline.pt"
    causal_path = models_dir / "causal.pt"
    if not synth_pred_path.exists():
        available = sorted(str(p.name) for p in models_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"Missing required synthetic predictor checkpoint: {synth_pred_path}\n"
            f"models_dir={models_dir} contains: {available}"
        )
    if not causal_path.exists():
        available = sorted(str(p.name) for p in models_dir.glob("*.pt"))
        raise FileNotFoundError(
            f"Missing required causal checkpoint: {causal_path}\n"
            f"models_dir={models_dir} contains: {available}\n\n"
            "To generate checkpoints, run:\n"
            "  uv run python -u examples/synthetic_corpus_generate.py --out-dir .data/synth_corpus\n"
            "  uv run python -u examples/synthetic_corpus_train.py --corpus-dir .data/synth_corpus --out-dir .data/synth_models\n"
        )

    synth_pred_ckpt = ModelCheckpoint.load(synth_pred_path)
    causal_ckpt = ModelCheckpoint.load(causal_path)
    if synth_pred_ckpt.model_type not in {"baseline", "l2"}:
        raise ValueError(
            f"Expected predictor checkpoint type baseline/l2, got {synth_pred_ckpt.model_type!r}"
        )
    if synth_pred_ckpt.arch not in {"deepsets_v1", "ft_transformer_v1", "mlp_legacy"}:
        raise ValueError(f"Unsupported predictor checkpoint arch={synth_pred_ckpt.arch!r}")
    if causal_ckpt.arch not in {"deepsets_v1", "ft_transformer_v1", "mlp_legacy"}:
        raise ValueError(f"Unsupported causal checkpoint arch={causal_ckpt.arch!r}")
    if synth_pred_ckpt.in_dim != causal_ckpt.in_dim:
        raise ValueError(
            "Predictor and causal checkpoints must share in_dim for fair comparison: "
            f"{synth_pred_ckpt.in_dim} vs {causal_ckpt.in_dim}"
        )
    in_dim = causal_ckpt.in_dim
    hidden_dim = causal_ckpt.hidden_dim
    model_kwargs = dict(causal_ckpt.model_kwargs)

    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    datasets = load_regression_datasets(cache_dir=args.cache_dir, names=dataset_names)

    rows: list[list[str]] = []
    per_regime_mae: dict[str, list[float]] = {
        "baseline_real": [],
        "synthetic_finetuned": [],
        "causal_finetuned": [],
    }
    per_regime_nmae: dict[str, list[float]] = {
        "baseline_real": [],
        "synthetic_finetuned": [],
        "causal_finetuned": [],
    }
    per_dataset_mae: dict[str, dict[str, float]] = {}
    per_dataset_d: dict[str, int] = {}
    per_dataset_d_eff: dict[str, int] = {}

    for ds in datasets:
        n, d = ds.X.shape
        d_eff = min(d, in_dim)
        if d > in_dim:
            print(
                f"[warn] dataset={ds.name}: D={d} exceeds checkpoint in_dim={in_dim}; "
                f"using first {d_eff} features for both regimes."
            )
        train_idx, test_idx = _train_test_split(n, test_frac=args.test_frac, seed=args.seed)
        tr_idx, val_idx = _train_val_split(train_idx, val_frac=args.val_frac, seed=args.seed + 1)

        if not (0.0 < args.train_frac <= 1.0):
            raise ValueError("--train-frac must be in (0,1]")
        if args.train_frac < 1.0:
            rng = np.random.default_rng(args.seed + 2)
            tr_perm = np.array(tr_idx, copy=True)
            rng.shuffle(tr_perm)
            n_sub = max(32, int(round(args.train_frac * len(tr_perm))))
            tr_idx = tr_perm[:n_sub]

        # Standardize using *train* only.
        mean, std = _standardize_fit(ds.X[tr_idx])
        X_tr = _standardize(ds.X[tr_idx], mean, std)
        X_val = _standardize(ds.X[val_idx], mean, std)
        X_te = _standardize(ds.X[test_idx], mean, std)
        y_tr = ds.y[tr_idx]
        y_val = ds.y[val_idx]
        y_te = ds.y[test_idx]

        x_tr, m_tr = _to_padded_tensor(X_tr, in_dim=in_dim)
        x_val, m_val = _to_padded_tensor(X_val, in_dim=in_dim)
        x_te, m_te = _to_padded_tensor(X_te, in_dim=in_dim)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.float32)

        # 1) Baseline trained on real
        baseline_real, y_mean_b, y_std_b = _train_baseline_on_real(
            x_train=x_tr,
            mask_train=m_tr,
            y_train=y_tr_t,
            x_val=x_val,
            mask_val=m_val,
            y_val=y_val_t,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            epochs=args.real_epochs,
            batch_size=args.real_batch_size,
            lr=args.real_lr,
            weight_decay=args.real_weight_decay,
            model_kwargs=model_kwargs,
            seed=args.seed,
        )

        # 2) Causal model pre-trained on synthetic, fine-tuned on real
        synthetic_ft, y_mean_s, y_std_s = _finetune_predictor_auto(
            ckpt=synth_pred_ckpt,
            x_train=x_tr,
            mask_train=m_tr,
            y_train=y_tr_t,
            x_val=x_val,
            mask_val=m_val,
            y_val=y_val_t,
            batch_size=args.real_batch_size,
            linear_probe_epochs=args.probe_epochs,
            finetune_epochs=args.finetune_epochs,
            lr_probe=args.probe_lr,
            lr_finetune=args.finetune_lr,
            lambda_sp=args.lambda_sp,
            seed=args.seed,
        )

        # 3) Causal model pre-trained on synthetic, fine-tuned on real
        if str(args.finetune_mode) == "auto":
            causal_ft, y_mean_c, y_std_c = _finetune_causal_auto(
                ckpt=causal_ckpt,
                x_train=x_tr,
                mask_train=m_tr,
                y_train=y_tr_t,
                x_val=x_val,
                mask_val=m_val,
                y_val=y_val_t,
                linear_probe_epochs=args.probe_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.real_batch_size,
                lr_probe=args.probe_lr,
                lr_finetune=args.finetune_lr,
                lambda_consistency=args.lambda_consistency,
                lambda_cb=args.lambda_cb,
                lambda_sp=args.lambda_sp,
                use_consistency=bool(args.use_consistency),
                freeze_ce_cb=not bool(args.no_freeze_ce_cb),
                seed=args.seed,
            )
        else:
            causal_ft, y_mean_c, y_std_c = _finetune_causal_on_real(
                ckpt=causal_ckpt,
                x_train=x_tr,
                mask_train=m_tr,
                y_train=y_tr_t,
                x_val=x_val,
                mask_val=m_val,
                y_val=y_val_t,
                linear_probe_epochs=args.probe_epochs,
                finetune_epochs=args.finetune_epochs,
                batch_size=args.real_batch_size,
                lr_probe=args.probe_lr,
                lr_finetune=args.finetune_lr,
                lambda_consistency=args.lambda_consistency,
                lambda_cb=args.lambda_cb,
                lambda_sp=args.lambda_sp,
                use_consistency=bool(args.use_consistency),
                freeze_ce_cb=not bool(args.no_freeze_ce_cb),
                finetune_mode=str(args.finetune_mode),
                seed=args.seed,
            )

        # Evaluate
        # Evaluate in original y units (unstandardize predictions).
        with torch.no_grad():
            pred_b_std = baseline_real(x_te, m_te)
            pred_b = pred_b_std * y_std_b + y_mean_b
            mae_b = float((pred_b - y_te_t).abs().mean().item())

            pred_s_std = synthetic_ft(x_te, m_te)
            pred_s = pred_s_std * y_std_s + y_mean_s
            mae_s = float((pred_s - y_te_t).abs().mean().item())

            pred_c_std, _ce, _cb = causal_ft(x_te, m_te)
            pred_c = pred_c_std * y_std_c + y_mean_c
            mae_c = float((pred_c - y_te_t).abs().mean().item())
        y_te_scale = float(y_te_t.std().clamp_min(1e-12).item())
        nmae_b = mae_b / y_te_scale
        nmae_s = mae_s / y_te_scale
        nmae_c = mae_c / y_te_scale
        per_regime_mae["baseline_real"].append(mae_b)
        per_regime_mae["synthetic_finetuned"].append(mae_s)
        per_regime_mae["causal_finetuned"].append(mae_c)
        per_regime_nmae["baseline_real"].append(nmae_b)
        per_regime_nmae["synthetic_finetuned"].append(nmae_s)
        per_regime_nmae["causal_finetuned"].append(nmae_c)
        per_dataset_mae[ds.name] = {
            "baseline_real": mae_b,
            "synthetic_finetuned": mae_s,
            "causal_finetuned": mae_c,
        }
        per_dataset_d[ds.name] = int(d)
        per_dataset_d_eff[ds.name] = int(d_eff)
        rows.append([ds.name, f"{d} ({d_eff})", "baseline_real", f"{mae_b:.6f}"])
        rows.append([ds.name, f"{d} ({d_eff})", "synthetic_finetuned", f"{mae_s:.6f}"])
        rows.append([ds.name, f"{d} ({d_eff})", "causal_finetuned", f"{mae_c:.6f}"])

        # Causal effect report (mean over test set)
        ce_rows: list[list[str]] = []
        for regime, model in [
            ("baseline_real", baseline_real),
            ("synthetic_finetuned", synthetic_ft),
            ("causal_finetuned", causal_ft),
        ]:
            ce_norm, cb_norm, ce_raw, cb_raw = _estimate_effects(
                model, x_te, m_te, feature_std=std.squeeze(0)
            )
            # Convert effects back to original y units.
            if regime == "baseline_real":
                ce_norm = ce_norm * y_std_b
                cb_norm = cb_norm * y_std_b
                ce_raw = ce_raw * y_std_b
                cb_raw = cb_raw * y_std_b
            elif regime == "synthetic_finetuned":
                ce_norm = ce_norm * y_std_s
                cb_norm = cb_norm * y_std_s
                ce_raw = ce_raw * y_std_s
                cb_raw = cb_raw * y_std_s
            else:
                ce_norm = ce_norm * y_std_c
                cb_norm = cb_norm * y_std_c
                ce_raw = ce_raw * y_std_c
                cb_raw = cb_raw * y_std_c
            dd = min(d, in_dim)
            ce_norm_str = ",".join(f"{v:.3f}" for v in ce_norm[:dd])
            cb_norm_str = ",".join(f"{v:.3f}" for v in cb_norm[:dd])
            ce_raw_str = ",".join(f"{v:.3f}" for v in ce_raw[:dd])
            cb_raw_str = ",".join(f"{v:.3f}" for v in cb_raw[:dd])
            ce_rows.append(
                [ds.name, regime, ce_norm_str, cb_norm_str, ce_raw_str, cb_raw_str]
            )

        print_table(
            title=f"Estimated effects (dataset={ds.name})",
            headers=[
                "dataset",
                "regime",
                "ce_mean_per_1stdX",
                "cb_mean_per_1stdX",
                "ce_mean_rawX",
                "cb_mean_rawX",
            ],
            rows=ce_rows,
        )

    print_table(
        title="Prediction MAE on real regression datasets",
        headers=["dataset", "D (effective)", "regime", "test_mae"],
        rows=rows,
    )

    # Scale-normalized MAE (divide by test target std per dataset) for
    # cross-dataset comparability.
    nrows: list[list[str]] = []
    for name in [ds.name for ds in datasets]:
        # recover per-dataset metrics from the aligned rows
        b = per_dataset_mae[name]["baseline_real"]
        s = per_dataset_mae[name]["synthetic_finetuned"]
        c = per_dataset_mae[name]["causal_finetuned"]
        # infer the y scale from stored means: nmae = mae / y_scale.
        # use baseline entries to recover per-dataset y scale robustly.
        idx = [ds.name for ds in datasets].index(name)
        y_scale = b / max(per_regime_nmae["baseline_real"][idx], 1e-12)
        nrows.append([name, f"{per_dataset_d[name]} ({per_dataset_d_eff[name]})", "baseline_real", f"{(b / y_scale):.6f}"])
        nrows.append([name, f"{per_dataset_d[name]} ({per_dataset_d_eff[name]})", "synthetic_finetuned", f"{(s / y_scale):.6f}"])
        nrows.append([name, f"{per_dataset_d[name]} ({per_dataset_d_eff[name]})", "causal_finetuned", f"{(c / y_scale):.6f}"])
    print_table(
        title="Prediction NMAE on real regression datasets (MAE / std(y_test))",
        headers=["dataset", "D (effective)", "regime", "test_nmae"],
        rows=nrows,
    )

    # A clearer per-dataset comparison table.
    compare_rows: list[list[str]] = []
    deltas: list[float] = []
    for name in [ds.name for ds in datasets]:
        b = per_dataset_mae[name]["baseline_real"]
        s = per_dataset_mae[name]["synthetic_finetuned"]
        c = per_dataset_mae[name]["causal_finetuned"]
        delta_bs = s - b
        delta_bc = c - b
        delta_sc = c - s
        deltas.append(delta_bc)
        compare_rows.append(
            [
                name,
                f"{per_dataset_d[name]} ({per_dataset_d_eff[name]})",
                f"{b:.6f}",
                f"{s:.6f}",
                f"{c:.6f}",
                f"{delta_bs:+.6f}",
                f"{delta_bc:+.6f}",
                f"{delta_sc:+.6f}",
            ]
        )
    print_table(
        title="Per-dataset MAE comparison (lower is better; deltas are row-wise differences)",
        headers=[
            "dataset",
            "D (effective)",
            "baseline_mae",
            "synthetic_ft_mae",
            "causal_ft_mae",
            "delta(synth-baseline)",
            "delta(causal-baseline)",
            "delta(causal-synth)",
        ],
        rows=compare_rows,
    )

    summary_rows: list[list[str]] = []
    for regime in ["baseline_real", "synthetic_finetuned", "causal_finetuned"]:
        s = summary(per_regime_mae[regime])
        summary_rows.append(
            [regime, f"{s['mean']:.6f}", f"{s['median']:.6f}", f"{s['std']:.6f}"]
        )
    print_table(
        title="Overall MAE summary (across datasets)",
        headers=["regime", "mean", "median", "std"],
        rows=summary_rows,
    )

    nsummary_rows: list[list[str]] = []
    for regime in ["baseline_real", "synthetic_finetuned", "causal_finetuned"]:
        s = summary(per_regime_nmae[regime])
        nsummary_rows.append(
            [regime, f"{s['mean']:.6f}", f"{s['median']:.6f}", f"{s['std']:.6f}"]
        )
    print_table(
        title="Overall NMAE summary (across datasets; scale-normalized)",
        headers=["regime", "mean", "median", "std"],
        rows=nsummary_rows,
    )

    delta_summary = summary(deltas)
    print_table(
        title="Overall delta summary (causal - baseline; negative means causal is better)",
        headers=["mean", "median", "std"],
        rows=[
            [
                f"{delta_summary['mean']:+.6f}",
                f"{delta_summary['median']:+.6f}",
                f"{delta_summary['std']:+.6f}",
            ]
        ],
    )

    causal_wins_vs_baseline = 0
    causal_wins_vs_synth = 0
    causal_best_of_three = 0
    for i in range(len(datasets)):
        b = per_regime_mae["baseline_real"][i]
        s = per_regime_mae["synthetic_finetuned"][i]
        c = per_regime_mae["causal_finetuned"][i]
        if c < b:
            causal_wins_vs_baseline += 1
        if c < s:
            causal_wins_vs_synth += 1
        if c <= min(b, s):
            causal_best_of_three += 1
    print(
        f"\nCausal wins vs baseline_real: {causal_wins_vs_baseline}/{len(datasets)} datasets"
    )
    print(
        f"Causal wins vs synthetic_finetuned: {causal_wins_vs_synth}/{len(datasets)} datasets"
    )
    print(f"Causal best of all 3 regimes: {causal_best_of_three}/{len(datasets)} datasets")


if __name__ == "__main__":
    main()
