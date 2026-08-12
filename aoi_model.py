from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from convnextv2 import load_convnextv2_tiny
from modules.c2_ffn_convnext_v1 import load_c2_ffn_convnext_v1


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def grl(x: torch.Tensor, coefficient: float) -> torch.Tensor:
    return GradientReversal.apply(x, coefficient)


class AOIMultiBranchModel(nn.Module):
    """
    共享学生骨干：
    - F16局部外观热力图
    - F32全局异常分数
    - 组件状态头
    - 几何回归头
    - real/synthetic域分类头
    """

    def __init__(
        self,
        student_checkpoint: str,
        component_slots: int = 8,
        geometry_dims: int = 6,
        local_top_ratio: float = 0.01,
        backbone_mode: str = "dense",
    ):
        super().__init__()
        if backbone_mode in ("dense", "standard"):
            self.backbone = load_convnextv2_tiny(student_checkpoint)
        elif backbone_mode in ("c2_hard_b", "c2_ffn_v1_b"):
            self.backbone = load_c2_ffn_convnext_v1(
                checkpoint=student_checkpoint,
                proposal_hws=((2, 4), (2, 4), (2, 4)),
            )
        else:
            raise ValueError(f"Unknown backbone_mode: {backbone_mode}")
        self.backbone_mode = backbone_mode
        self.local_top_ratio = local_top_ratio

        self.local_head = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(192, 1, kernel_size=1),
        )

        self.global_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        self.component_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, component_slots),
        )

        self.geometry_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, geometry_dims),
        )

        self.domain_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        domain_coefficient: float = 0.0,
        external_scores: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.backbone(image)
        f16 = features["f16"]
        f32 = features["f32"]

        local_logits = self.local_head(f16)
        local_flat = local_logits.flatten(1)
        k = max(
            1,
            int(round(local_flat.shape[1] * self.local_top_ratio)),
        )
        local_top = torch.topk(
            local_flat, k=k, dim=1
        ).values.mean(dim=1)
        local_mean = local_flat.mean(dim=1)

        global_feature = f32.mean(dim=(-2, -1))
        global_logit = self.global_head(
            global_feature
        ).squeeze(-1)

        component_logits = self.component_head(global_feature)
        component_uncertainty = (
            torch.sigmoid(component_logits)
            * (1.0 - torch.sigmoid(component_logits))
        ).mean(dim=1)

        geometry = self.geometry_head(global_feature)
        geometry_magnitude = geometry.abs().mean(dim=1)

        if external_scores is None:
            external_scores = torch.zeros(
                image.shape[0],
                2,
                device=image.device,
                dtype=image.dtype,
            )

        fusion_input = torch.stack(
            [
                local_top,
                local_mean,
                global_logit,
                component_uncertainty,
                geometry_magnitude,
                external_scores.mean(dim=1),
            ],
            dim=1,
        )

        final_logit = (
            torch.maximum(local_top, global_logit)
            + 0.5 * torch.tanh(
                self.fusion_head(fusion_input).squeeze(-1)
            )
        )

        domain_logit = self.domain_head(
            grl(global_feature, domain_coefficient)
        ).squeeze(-1)

        return {
            "final_logit": final_logit,
            "local_logits": local_logits,
            "global_logit": global_logit,
            "component_logits": component_logits,
            "geometry": geometry,
            "domain_logit": domain_logit,
            "features": features,
        }


class TemporalLogicHead(nn.Module):
    """视频顺序/状态分支。输入每帧的多分支统计。"""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_states: int = 8,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.state_head = nn.Linear(hidden_dim, num_states)
        self.sequence_anomaly_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor):
        hidden, _ = self.gru(sequence)
        states = self.state_head(hidden)
        sequence_score = self.sequence_anomaly_head(
            hidden[:, -1]
        ).squeeze(-1)
        return {
            "state_logits": states,
            "sequence_logit": sequence_score,
        }
