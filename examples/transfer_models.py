"""Models and helpers for synthetic-to-real transfer benchmarks.

Key design goal: **transferability across datasets with unrelated feature semantics**.

Plain MLPs over a fixed feature order tend to overfit to positional/semantic
assumptions that do not hold across unrelated real datasets. To make synthetic
pretraining useful, we use *set / token* models that are permutation-invariant
and can model interactions:

- `DeepSets` (fast, but limited interactions)
- `FeatureTransformer` (self-attention over features, more expressive)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    num_hidden_layers: int = 1,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DeepSetsPredictor(nn.Module):
    """Permutation-invariant predictor.

    Treats each scalar feature as an element in a set and aggregates via mean-pooling.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        *,
        feature_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.feature_embed_dim = feature_embed_dim or hidden_dim

        # phi: R -> R^H
        self.feature_encoder = _mlp(1, hidden_dim, self.feature_embed_dim, num_hidden_layers=2)
        # rho: R^H -> R
        self.aggregator = _mlp(self.feature_embed_dim, hidden_dim, 1, num_hidden_layers=2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B,D), mask: (B,D)
        _b, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected x with D={self.in_dim}, got {d}")

        h = self.feature_encoder(x.unsqueeze(-1))  # (B,D,H)
        h = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = h.sum(dim=1) / denom  # (B,H)
        return self.aggregator(pooled).squeeze(-1)


class DeepSetsCausalConsistencyModel(nn.Module):
    """Permutation-invariant causal-consistency model (DeepSets variant)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        *,
        feature_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if out_dim != in_dim:
            raise ValueError("out_dim must equal in_dim")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.feature_embed_dim = feature_embed_dim or hidden_dim

        self.feature_encoder = _mlp(1, hidden_dim, self.feature_embed_dim, num_hidden_layers=2)
        self.pred_head = _mlp(self.feature_embed_dim, hidden_dim, 1, num_hidden_layers=2)
        self.per_feature_head = _mlp(
            self.feature_embed_dim + self.feature_embed_dim,
            hidden_dim,
            2,
            num_hidden_layers=2,
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected x with D={self.in_dim}, got {d}")

        h = self.feature_encoder(x.unsqueeze(-1))  # (B,D,H)
        h_masked = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = h_masked.sum(dim=1) / denom  # (B,H)

        pred = self.pred_head(pooled).squeeze(-1)
        pooled_exp = pooled.unsqueeze(1).expand(b, d, pooled.shape[-1])
        z = torch.cat([h, pooled_exp], dim=-1)
        ce_cb = self.per_feature_head(z)  # (B,D,2)
        ce = ce_cb[..., 0] * mask
        cb = ce_cb[..., 1] * mask
        return pred, ce, cb


class FeatureTransformerBackbone(nn.Module):
    """Self-attention backbone over feature tokens."""

    def __init__(
        self,
        *,
        max_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        norm_first: bool = False,
        adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward or 4 * d_model

        # Feature-id embeddings break the exchangeability symmetry and allow the
        # model to represent feature-specific roles within a dataset.
        # We still get transfer because during synthetic pretraining we permute
        # feature columns (and thus ids) per task.
        self.feature_id_embed = nn.Embedding(max_features, d_model)

        self.value_embed = nn.Linear(1, d_model)
        # NOTE: keep norm_first=False to avoid PyTorch's warning about nested
        # tensors being disabled when norm_first=True.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=norm_first,
        )
        # Disable nested tensor fast-path to avoid noisy prototype-stage warnings
        # in recent PyTorch versions.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Optional small residual adapter for parameter-efficient fine-tuning.
        # When adapter_dim>0, you can freeze the encoder and only train this
        # adapter + the prediction head during few-shot transfer.
        self.adapter_dim = int(adapter_dim)
        if self.adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.adapter_dim),
                nn.ReLU(),
                nn.Linear(self.adapter_dim, d_model),
            )
        else:
            self.adapter = None

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B,D), mask: (B,D)
        b, d = x.shape
        tokens = self.value_embed(x.unsqueeze(-1))  # (B,D,d_model)
        idx = torch.arange(d, device=x.device)
        tokens = tokens + self.feature_id_embed(idx).unsqueeze(0).expand(b, d, -1)
        # Transformer expects True for positions that should be masked.
        key_padding_mask = mask <= 0.0  # (B,D)
        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        if self.adapter is not None:
            out = out + self.adapter(out)
        return out


class FeatureTransformerPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        *,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm_first: bool = False,
        adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.backbone = FeatureTransformerBackbone(
            max_features=in_dim,
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            norm_first=norm_first,
            adapter_dim=adapter_dim,
        )
        self.pred_head = _mlp(hidden_dim, hidden_dim, 1, num_hidden_layers=2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        _b, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected x with D={self.in_dim}, got {d}")

        h = self.backbone(x, mask)  # (B,D,H)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.pred_head(pooled).squeeze(-1)


class FeatureTransformerCausalConsistencyModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        *,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        norm_first: bool = False,
        adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        if out_dim != in_dim:
            raise ValueError("out_dim must equal in_dim")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.backbone = FeatureTransformerBackbone(
            max_features=in_dim,
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            norm_first=norm_first,
            adapter_dim=adapter_dim,
        )
        self.pred_head = _mlp(hidden_dim, hidden_dim, 1, num_hidden_layers=2)
        self.per_feature_head = _mlp(hidden_dim + hidden_dim, hidden_dim, 2, num_hidden_layers=2)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected x with D={self.in_dim}, got {d}")

        h = self.backbone(x, mask)  # (B,D,H)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)
        pred = self.pred_head(pooled).squeeze(-1)

        pooled_exp = pooled.unsqueeze(1).expand(b, d, pooled.shape[-1])
        z = torch.cat([h, pooled_exp], dim=-1)
        ce_cb = self.per_feature_head(z)  # (B,D,2)
        ce = ce_cb[..., 0] * mask
        cb = ce_cb[..., 1] * mask
        return pred, ce, cb


Predictor = FeatureTransformerPredictor


CausalConsistencyModel = FeatureTransformerCausalConsistencyModel


@dataclass(frozen=True)
class ModelCheckpoint:
    model_type: str  # "baseline" | "l2" | "causal"
    arch: str  # "deepsets_v1" | "ft_transformer_v1" | "mlp_legacy"
    in_dim: int
    out_dim: int | None
    hidden_dim: int
    state_dict: dict[str, Any]
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": self.model_type,
                "arch": self.arch,
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "hidden_dim": self.hidden_dim,
                "model_kwargs": self.model_kwargs,
                "state_dict": self.state_dict,
            },
            p,
        )

    @staticmethod
    def load(path: str | Path) -> "ModelCheckpoint":
        blob = torch.load(Path(path), map_location="cpu")
        return ModelCheckpoint(
            model_type=str(blob["model_type"]),
            arch=str(blob.get("arch", "mlp_legacy")),
            in_dim=int(blob["in_dim"]),
            out_dim=None if blob["out_dim"] is None else int(blob["out_dim"]),
            hidden_dim=int(blob["hidden_dim"]),
            model_kwargs=dict(blob.get("model_kwargs", {})),
            state_dict=dict(blob["state_dict"]),
        )


def build_model_from_checkpoint(
    ckpt: ModelCheckpoint,
) -> nn.Module:
    kwargs = dict(ckpt.model_kwargs)
    if ckpt.arch == "deepsets_v1":
        if ckpt.model_type in {"baseline", "l2"}:
            model = DeepSetsPredictor(in_dim=ckpt.in_dim, hidden_dim=ckpt.hidden_dim, **kwargs)
        elif ckpt.model_type == "causal":
            if ckpt.out_dim is None:
                raise ValueError("causal checkpoint must have out_dim")
            model = DeepSetsCausalConsistencyModel(
                in_dim=ckpt.in_dim, out_dim=ckpt.out_dim, hidden_dim=ckpt.hidden_dim, **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {ckpt.model_type}")
    elif ckpt.arch in {"ft_transformer_v1", "mlp_legacy"}:
        # Predictor/CausalConsistencyModel aliases currently point to FT-Transformer.
        if ckpt.model_type in {"baseline", "l2"}:
            model = Predictor(in_dim=ckpt.in_dim, hidden_dim=ckpt.hidden_dim, **kwargs)
        elif ckpt.model_type == "causal":
            if ckpt.out_dim is None:
                raise ValueError("causal checkpoint must have out_dim")
            model = CausalConsistencyModel(
                in_dim=ckpt.in_dim, out_dim=ckpt.out_dim, hidden_dim=ckpt.hidden_dim, **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {ckpt.model_type}")
    else:
        raise ValueError(f"Unsupported checkpoint arch: {ckpt.arch}")
    # Backward-compatibility: older checkpoints (created before feature-id
    # embeddings were introduced in FeatureTransformerBackbone) do not contain
    # `backbone.feature_id_embed.weight`.
    #
    # In that case, keep the module's random initialization for this embedding
    # and load the rest of the weights non-strictly.
    try:
        model.load_state_dict(ckpt.state_dict)
    except RuntimeError as err:
        missing, unexpected = model.load_state_dict(ckpt.state_dict, strict=False)
        allowed_missing = {"backbone.feature_id_embed.weight"}
        if not set(unexpected) == set() or not set(missing).issubset(allowed_missing):
            raise err
    model.eval()
    return model
