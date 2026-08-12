from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HardRoute:
    cluster_idx: torch.Tensor
    aggregate_weight: torch.Tensor
    clusters: int
    height: int
    width: int


class RegionHardRouter(nn.Module):
    """Build one hard routing map per stage and reuse it in every block."""

    def __init__(
        self,
        fold_hw: Tuple[int, int],
        proposal_hw: Tuple[int, int] = (2, 2),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.fold_hw = tuple(int(value) for value in fold_hw)
        self.proposal_hw = tuple(int(value) for value in proposal_hw)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> HardRoute:
        batch, channels, height, width = x.shape
        fold_h, fold_w = self.fold_hw
        proposal_h, proposal_w = self.proposal_hw
        region_h = (height + fold_h - 1) // fold_h
        region_w = (width + fold_w - 1) // fold_w
        padded_h, padded_w = region_h * fold_h, region_w * fold_w
        if padded_h != height or padded_w != width:
            x = F.pad(
                x,
                (0, padded_w - width, 0, padded_h - height),
                mode="replicate",
            )

        regions = (
            x.reshape(batch, channels, fold_h, region_h, fold_w, region_w)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch * fold_h * fold_w, channels, region_h, region_w)
        )
        proposals = F.adaptive_avg_pool2d(regions, (proposal_h, proposal_w))
        pixels = regions.flatten(2).transpose(1, 2)
        centers = proposals.flatten(2).transpose(1, 2)
        similarity = torch.bmm(
            F.normalize(pixels.float(), dim=-1, eps=self.eps),
            F.normalize(centers.float(), dim=-1, eps=self.eps).transpose(1, 2),
        )
        top_similarity, local_idx = similarity.max(dim=-1)

        centers_per_region = proposal_h * proposal_w
        region_count = fold_h * fold_w
        region_id = torch.arange(
            region_count, device=x.device, dtype=local_idx.dtype
        ).view(1, region_count, 1)
        local_idx = local_idx.reshape(batch, region_count, region_h * region_w)
        global_idx = local_idx + region_id * centers_per_region
        confidence = 0.5 + 0.5 * top_similarity.clamp(0.0, 1.0)
        aggregate_weight = confidence.reshape(
            batch, region_count, region_h * region_w
        ) * (float(centers_per_region) / float(region_h * region_w))

        def restore(values: torch.Tensor) -> torch.Tensor:
            values = values.reshape(
                batch, fold_h, fold_w, region_h, region_w
            ).permute(0, 1, 3, 2, 4)
            return values.reshape(batch, padded_h, padded_w)[:, :height, :width]

        return HardRoute(
            cluster_idx=restore(global_idx).reshape(batch, height * width),
            aggregate_weight=restore(aggregate_weight)
            .reshape(batch, height * width)
            .to(dtype=x.dtype),
            clusters=region_count * centers_per_region,
            height=height,
            width=width,
        )


class ClusterFFNBlock(nn.Module):
    """Keep dense DWConv/LN, but execute the full channel FFN on centroids."""

    def __init__(self, original_block: nn.Module):
        super().__init__()
        self.dwconv = original_block.dwconv
        self.norm = original_block.norm
        self.pwconv1 = original_block.pwconv1
        self.act = original_block.act
        self.grn = original_block.grn
        self.pwconv2 = original_block.pwconv2

    @staticmethod
    def aggregate(z: torch.Tensor, route: HardRoute) -> torch.Tensor:
        batch, height, width, channels = z.shape
        flat = z.reshape(batch, height * width, channels)
        index = route.cluster_idx.unsqueeze(-1).expand(-1, -1, channels)
        weight = route.aggregate_weight.unsqueeze(-1).to(dtype=z.dtype)
        numerator = torch.zeros(
            batch, route.clusters, channels, device=z.device, dtype=z.dtype
        )
        numerator.scatter_add_(1, index, flat * weight)
        mass = torch.zeros(
            batch, route.clusters, 1, device=z.device, dtype=z.dtype
        )
        mass.scatter_add_(1, route.cluster_idx.unsqueeze(-1), weight)
        return numerator / mass.clamp_min(1e-4)

    @staticmethod
    def diffuse(delta: torch.Tensor, route: HardRoute) -> torch.Tensor:
        channels = delta.shape[-1]
        index = route.cluster_idx.unsqueeze(-1).expand(-1, -1, channels)
        dense = torch.gather(delta, 1, index)
        return dense.reshape(
            delta.shape[0], route.height, route.width, channels
        )

    def forward(self, x: torch.Tensor, route: HardRoute) -> torch.Tensor:
        residual = x
        z = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
        centroid = self.aggregate(z, route)
        delta = self.pwconv1(centroid).unsqueeze(2)
        delta = self.pwconv2(self.grn(self.act(delta))).squeeze(2)
        dense_delta = self.diffuse(delta, route).permute(0, 3, 1, 2)
        return (residual + dense_delta).contiguous(
            memory_format=torch.channels_last
        )


class C2FFNStage(nn.Module):
    def __init__(
        self,
        original_stage: nn.Sequential,
        fold_hw: Tuple[int, int],
        proposal_hw: Tuple[int, int],
    ):
        super().__init__()
        self.router = RegionHardRouter(fold_hw, proposal_hw)
        self.blocks = nn.ModuleList(
            [ClusterFFNBlock(block) for block in original_stage]
        )
        self.fold_hw = fold_hw
        self.proposal_hw = proposal_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = self.router(x)
        for block in self.blocks:
            x = block(x, route)
        return x


class C2FFNConvNeXtV1(nn.Module):
    """Hard-B C2-FFN ConvNeXt: clustered stages 1-3, dense stage 4."""

    def __init__(
        self,
        backbone: nn.Module,
        fold_hws: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
        proposal_hws: Sequence[Tuple[int, int]] = ((2, 4), (2, 4), (2, 4)),
    ):
        super().__init__()
        if len(fold_hws) != 3 or len(proposal_hws) != 3:
            raise ValueError("stage settings must contain three entries")
        self.downsample_layers = backbone.downsample_layers
        self.c2_stages = nn.ModuleList(
            [
                C2FFNStage(backbone.stages[i], fold_hws[i], proposal_hws[i])
                for i in range(3)
            ]
        )
        self.stage4 = backbone.stages[3]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        x = x.contiguous(memory_format=torch.channels_last)
        for index in range(3):
            x = self.downsample_layers[index](x)
            x = self.c2_stages[index](
                x.contiguous(memory_format=torch.channels_last)
            )
            outputs[("f4", "f8", "f16")[index]] = x
        x = self.downsample_layers[3](x)
        x = self.stage4(x.contiguous(memory_format=torch.channels_last))
        outputs["f32"] = x.contiguous(memory_format=torch.channels_last)
        return outputs


def load_c2_ffn_convnext_v1(
    checkpoint: str,
    proposal_hws: Sequence[Tuple[int, int]] = ((2, 4), (2, 4), (2, 4)),
) -> C2FFNConvNeXtV1:
    from convnextv2 import load_convnextv2_tiny

    return C2FFNConvNeXtV1(
        load_convnextv2_tiny(checkpoint),
        proposal_hws=proposal_hws,
    )
