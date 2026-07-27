from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )


class GRN(nn.Module):
    """ConvNeXt V2 Global Response Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtV2Tiny(nn.Module):
    """ConvNeXt V2 Tiny: depths=[3,3,9,3], dims=[96,192,384,768]."""

    def __init__(
        self,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
    ):
        super().__init__()
        self.dims = dims

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(
                    dims[0],
                    eps=1e-6,
                    data_format="channels_first",
                ),
            )
        )

        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(
                        dims[index],
                        eps=1e-6,
                        data_format="channels_first",
                    ),
                    nn.Conv2d(
                        dims[index],
                        dims[index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    *(Block(dims[i]) for _ in range(depths[i]))
                )
                for i in range(4)
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        names = ("f4", "f8", "f16", "f32")

        for index in range(4):
            x = self.downsample_layers[index](x)
            x = self.stages[index](x)
            outputs[names[index]] = x

        return outputs


def _unwrap_state_dict(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict-like object.")

    for candidate in (
        "model_ema",
        "model",
        "state_dict",
        "module",
        "encoder",
    ):
        value = checkpoint.get(candidate)
        if isinstance(value, dict):
            checkpoint = value
            break

    return checkpoint


def _normalize_key(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "backbone.",
        "encoder.",
        "student.",
    )

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def load_convnextv2_tiny(
    checkpoint_path: str | Path,
    map_location: str = "cpu",
) -> ConvNeXtV2Tiny:
    model = ConvNeXtV2Tiny()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state = _unwrap_state_dict(checkpoint)

    cleaned = {}
    for key, value in state.items():
        normalized = _normalize_key(key)

        if normalized.startswith(
            ("head.", "norm.", "decoder.", "mask_token")
        ):
            continue

        cleaned[normalized] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    # 允许分类头相关缺失，但骨干大规模缺失通常意味着权重格式不匹配。
    backbone_missing = [
        key
        for key in missing
        if not key.startswith(("head.", "norm."))
    ]

    if len(backbone_missing) > 8:
        preview = "\n".join(backbone_missing[:20])
        raise RuntimeError(
            "ConvNeXtV2-Tiny权重未正确加载。"
            f"\nMissing count={len(backbone_missing)}"
            f"\nFirst missing keys:\n{preview}"
        )

    if unexpected:
        print(
            f"[ConvNeXtV2] ignored unexpected keys: {len(unexpected)}"
        )

    print(
        f"[ConvNeXtV2] loaded. missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    return model
