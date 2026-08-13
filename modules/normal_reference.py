from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from config import AOIConfig
from aoi_model import AOIMultiBranchModel
from utils.image import (
    geometry_statistics,
    lab_statistics,
    load_rgb,
    normalize_image,
)


@dataclass
class ScoreCalibration:
    local_mean: float
    local_std: float
    global_mean: float
    global_std: float
    color_mean: float
    color_std: float
    geometry_mean: float
    geometry_std: float


class NormalReference:
    """100张正常图建立的局部、全局、颜色与几何参考。"""

    def __init__(self, config: AOIConfig):
        self.config = config
        self.local_bank: np.ndarray | None = None
        self.global_mean: np.ndarray | None = None
        self.global_precision: np.ndarray | None = None
        self.color_mean: np.ndarray | None = None
        self.color_precision: np.ndarray | None = None
        self.geometry_mean: np.ndarray | None = None
        self.geometry_precision: np.ndarray | None = None
        self.calibration: ScoreCalibration | None = None
        self.threshold: float | None = None
        self._local_bank_tensor = None
        self._local_bank_sq_norm = None
        self._local_bank_device = None

    @staticmethod
    def _mahalanobis(
        value: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
    ) -> float:
        difference = value - mean
        return float(difference @ precision @ difference)

    def _clear_local_bank_cache(self) -> None:
        self._local_bank_tensor = None
        self._local_bank_sq_norm = None
        self._local_bank_device = None

    @torch.inference_mode()
    def _extract(
        self,
        model: AOIMultiBranchModel,
        image_path: str | Path,
    ):
        image = load_rgb(image_path)
        tensor = normalize_image(
            image,
            self.config.global_size,
        )[None].to(self.config.device)

        output = model(tensor)
        f16 = (
            output["features"]["f16"][0]
            .permute(1, 2, 0)
            .reshape(-1, 384)
            .float()
            .cpu()
            .numpy()
        )
        f32 = (
            output["features"]["f32"]
            .mean(dim=(-2, -1))[0]
            .float()
            .cpu()
            .numpy()
        )

        return {
            "local": f16,
            "global": f32,
            "color": lab_statistics(image),
            "geometry": geometry_statistics(image),
        }

    def fit(
        self,
        model: AOIMultiBranchModel,
        normal_paths: Iterable[str | Path],
    ) -> None:
        model.eval()
        rng = np.random.default_rng(42)

        local_tokens = []
        global_features = []
        color_features = []
        geometry_features = []

        normal_paths = list(normal_paths)
        if len(normal_paths) < 5:
            raise ValueError("至少需要5张正常图建立参考。")

        for path in normal_paths:
            features = self._extract(model, path)

            tokens = features["local"]
            count = min(
                self.config.local_tokens_per_image,
                len(tokens),
            )
            indices = rng.choice(
                len(tokens), size=count, replace=False
            )
            local_tokens.append(tokens[indices])
            global_features.append(features["global"])
            color_features.append(features["color"])
            geometry_features.append(features["geometry"])

        bank = np.concatenate(local_tokens, axis=0)
        if len(bank) > self.config.max_local_tokens:
            indices = rng.choice(
                len(bank),
                size=self.config.max_local_tokens,
                replace=False,
            )
            bank = bank[indices]
        self.local_bank = bank.astype(np.float32)
        self._clear_local_bank_cache()

        global_array = np.stack(global_features)
        global_model = LedoitWolf().fit(global_array)
        self.global_mean = global_model.location_.astype(np.float32)
        self.global_precision = global_model.precision_.astype(
            np.float32
        )

        color_array = np.stack(color_features)
        color_model = LedoitWolf().fit(color_array)
        self.color_mean = color_model.location_.astype(np.float32)
        self.color_precision = color_model.precision_.astype(
            np.float32
        )

        geometry_array = np.stack(geometry_features)
        geometry_model = LedoitWolf().fit(geometry_array)
        self.geometry_mean = geometry_model.location_.astype(
            np.float32
        )
        self.geometry_precision = geometry_model.precision_.astype(
            np.float32
        )

        raw_scores = [
            self.score_features(
                local=tokens,
                global_feature=global_feature,
                color=color,
                geometry=geometry,
            )
            for tokens, global_feature, color, geometry in zip(
                [item for item in local_tokens],
                global_features,
                color_features,
                geometry_features,
            )
        ]

        local_scores = np.array(
            [item["memory_local"] for item in raw_scores]
        )
        global_scores = np.array(
            [item["memory_global"] for item in raw_scores]
        )
        color_scores = np.array(
            [item["color"] for item in raw_scores]
        )
        geometry_scores = np.array(
            [item["geometry"] for item in raw_scores]
        )

        self.calibration = ScoreCalibration(
            local_mean=float(local_scores.mean()),
            local_std=float(local_scores.std() + 1e-6),
            global_mean=float(global_scores.mean()),
            global_std=float(global_scores.std() + 1e-6),
            color_mean=float(color_scores.mean()),
            color_std=float(color_scores.std() + 1e-6),
            geometry_mean=float(geometry_scores.mean()),
            geometry_std=float(geometry_scores.std() + 1e-6),
        )

    def score_features(
        self,
        local: np.ndarray,
        global_feature: np.ndarray,
        color: np.ndarray,
        geometry: np.ndarray,
    ) -> dict:
        if self.local_bank is None:
            raise RuntimeError("NormalReference尚未fit。")

        minimum = np.full(len(local), np.inf, dtype=np.float32)

        for start in range(0, len(self.local_bank), 512):
            bank = self.local_bank[start:start + 512]
            distances = (
                (local[:, None, :] - bank[None, :, :]) ** 2
            ).mean(axis=2)
            minimum = np.minimum(minimum, distances.min(axis=1))

        k = max(1, int(round(len(minimum) * 0.01)))
        memory_local = float(np.sort(minimum)[-k:].mean())

        memory_global = self._mahalanobis(
            global_feature,
            self.global_mean,
            self.global_precision,
        )
        color_score = self._mahalanobis(
            color,
            self.color_mean,
            self.color_precision,
        )
        geometry_score = self._mahalanobis(
            geometry,
            self.geometry_mean,
            self.geometry_precision,
        )

        return {
            "memory_local": memory_local,
            "memory_global": memory_global,
            "color": color_score,
            "geometry": geometry_score,
            "local_map": minimum,
        }

    def zscore(self, scores: dict) -> dict:
        calibration = self.calibration
        if calibration is None:
            raise RuntimeError("缺少正常校准统计。")

        return {
            "memory_local": (
                scores["memory_local"] - calibration.local_mean
            ) / calibration.local_std,
            "memory_global": (
                scores["memory_global"] - calibration.global_mean
            ) / calibration.global_std,
            "color": (
                scores["color"] - calibration.color_mean
            ) / calibration.color_std,
            "geometry": (
                scores["geometry"] - calibration.geometry_mean
            ) / calibration.geometry_std,
            "local_map": scores.get(
                "local_map",
                np.empty(
                    0,
                    dtype=np.float32,
                ),
            ),
        }

    def _get_local_bank_tensor(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将正常局部Token库常驻GPU。

        返回：
        bank: [M, C]
        bank_sq_norm: [M]
        """
        if self.local_bank is None:
            raise RuntimeError(
                "局部正常Token库尚未建立。"
            )

        device_string = str(device)

        if (
            self._local_bank_tensor is None
            or self._local_bank_device
            != device_string
        ):
            bank = torch.as_tensor(
                self.local_bank,
                dtype=torch.float32,
                device=device,
            ).contiguous()

            self._local_bank_tensor = bank
            self._local_bank_sq_norm = (
                bank.square().sum(dim=1)
            )
            self._local_bank_device = (
                device_string
            )

        return (
            self._local_bank_tensor,
            self._local_bank_sq_norm,
        )


    @torch.inference_mode()
    def score_local_torch(
        self,
        local_tokens: torch.Tensor,
    ) -> float:
        """
        使用GPU矩阵乘法计算局部Token最近邻距离。

        与原始NumPy实现保持同一距离定义：

            mean((token - reference) ** 2)

        但不构造[N, M, C]三维广播数组。
        """
        if local_tokens.ndim != 2:
            raise ValueError(
                "local_tokens必须是[N, C]，"
                f"当前形状：{tuple(local_tokens.shape)}"
            )

        tokens = (
            local_tokens
            .detach()
            .to(dtype=torch.float32)
            .contiguous()
        )

        bank, bank_sq_norm = (
            self._get_local_bank_tensor(
                tokens.device
            )
        )

        if tokens.shape[1] != bank.shape[1]:
            raise ValueError(
                "当前Token与正常Token库维度不一致："
                f"{tokens.shape[1]} != {bank.shape[1]}"
            )

        token_sq_norm = (
            tokens.square()
            .sum(dim=1, keepdim=True)
        )

        minimum_distance = torch.full(
            (tokens.shape[0],),
            float("inf"),
            dtype=torch.float32,
            device=tokens.device,
        )

        chunk_size = int(
            getattr(
                self.config,
                "local_bank_chunk_size",
                4096,
            )
        )
        chunk_size = max(1, chunk_size)

        channel_count = float(
            tokens.shape[1]
        )

        for start in range(
            0,
            bank.shape[0],
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                bank.shape[0],
            )

            bank_chunk = bank[start:end]
            bank_norm_chunk = (
                bank_sq_norm[start:end]
            )

            # ||x-y||² = ||x||² + ||y||² - 2xy
            distances = (
                token_sq_norm
                + bank_norm_chunk.unsqueeze(0)
                - 2.0
                * torch.matmul(
                    tokens,
                    bank_chunk.transpose(0, 1),
                )
            )

            distances = (
                distances
                .clamp_min_(0.0)
                / channel_count
            )

            chunk_minimum = distances.min(
                dim=1
            ).values

            minimum_distance = torch.minimum(
                minimum_distance,
                chunk_minimum,
            )

        top_ratio = float(
            getattr(
                self.config,
                "local_top_ratio",
                0.01,
            )
        )

        top_count = max(
            1,
            int(
                round(
                    minimum_distance.numel()
                    * top_ratio
                )
            ),
        )

        local_score = torch.topk(
            minimum_distance,
            k=top_count,
            largest=True,
        ).values.mean()

        return float(
            local_score.detach().cpu()
        )

    # def score_reused_features(
    #     self,
    #     local_tokens: torch.Tensor,
    #     global_feature: np.ndarray,
    #     color: np.ndarray,
    #     geometry: np.ndarray,
    # ) -> dict:
    #     """
    #     直接使用当前模型前向得到的特征。

    #     不再次运行backbone。
    #     """
    #     raw_scores = {
    #         "memory_local": (
    #             self.score_local_torch(
    #                 local_tokens
    #             )
    #         ),
    #         "memory_global": (
    #             self._mahalanobis(
    #                 global_feature,
    #                 self.global_mean,
    #                 self.global_precision,
    #             )
    #         ),
    #         "color": self._mahalanobis(
    #             color,
    #             self.color_mean,
    #             self.color_precision,
    #         ),
    #         "geometry": self._mahalanobis(
    #             geometry,
    #             self.geometry_mean,
    #             self.geometry_precision,
    #         ),
    #     }

    #     return self.zscore(raw_scores)

    def score_fast_features(
        self,
        global_feature: np.ndarray,
        color: np.ndarray,
        geometry: np.ndarray,
    ) -> dict:
        """
        实时部署的快速正常参考评分。

        不执行：
        当前局部token × 12000正常token的CPU暴力最近邻。

        只计算：
        全局Mahalanobis、颜色统计、几何统计。
        """
        if self.calibration is None:
            raise RuntimeError(
                "NormalReference缺少校准统计。"
            )

        if (
            self.global_mean is None
            or self.global_precision is None
            or self.color_mean is None
            or self.color_precision is None
            or self.geometry_mean is None
            or self.geometry_precision is None
        ):
            raise RuntimeError(
                "NormalReference尚未完成fit。"
            )

        raw_global = self._mahalanobis(
            global_feature,
            self.global_mean,
            self.global_precision,
        )

        raw_color = self._mahalanobis(
            color,
            self.color_mean,
            self.color_precision,
        )

        raw_geometry = self._mahalanobis(
            geometry,
            self.geometry_mean,
            self.geometry_precision,
        )

        calibration = self.calibration

        return {
            # 局部检索关闭时保持中性贡献，而不是产生异常负值。
            "memory_local": 0.0,

            "memory_global": (
                raw_global
                - calibration.global_mean
            ) / calibration.global_std,

            "color": (
                raw_color
                - calibration.color_mean
            ) / calibration.color_std,

            "geometry": (
                raw_geometry
                - calibration.geometry_mean
            ) / calibration.geometry_std,

            "local_map": np.empty(
                0,
                dtype=np.float32,
            ),
        }

    @torch.inference_mode()
    def score_image(
        self,
        model: AOIMultiBranchModel,
        image_path: str | Path,
    ) -> dict:
        features = self._extract(model, image_path)
        return self.zscore(
            self.score_features(
                local=features["local"],
                global_feature=features["global"],
                color=features["color"],
                geometry=features["geometry"],
            )
        )

    @torch.inference_mode()
    def score_reused_features(
        self,
        local_tokens: torch.Tensor,
        global_feature: np.ndarray,
        color: np.ndarray,
        geometry: np.ndarray,
    ) -> dict:
        """
        复用当前推理前向得到的特征进行正常参考评分。

        不重新运行模型，只计算：
        1. GPU局部Token最近邻距离；
        2. 全局Mahalanobis距离；
        3. 颜色统计距离；
        4. 几何统计距离。
        """
        if self.calibration is None:
            raise RuntimeError(
                "NormalReference缺少calibration，"
                "请先调用fit()建立正常参考。"
            )

        if self.local_bank is None:
            raise RuntimeError(
                "NormalReference缺少local_bank，"
                "请先调用fit()建立正常参考。"
            )

        if (
            self.global_mean is None
            or self.global_precision is None
            or self.color_mean is None
            or self.color_precision is None
            or self.geometry_mean is None
            or self.geometry_precision is None
        ):
            raise RuntimeError(
                "NormalReference统计参数不完整，"
                "请重新运行目标迁移。"
            )

        raw_scores = {
            "memory_local": self.score_local_torch(
                local_tokens
            ),

            "memory_global": self._mahalanobis(
                global_feature,
                self.global_mean,
                self.global_precision,
            ),

            "color": self._mahalanobis(
                color,
                self.color_mean,
                self.color_precision,
            ),

            "geometry": self._mahalanobis(
                geometry,
                self.geometry_mean,
                self.geometry_precision,
            ),
            "local_map": np.empty(
                0,
                dtype=np.float32,
            ),
        }

        return self.zscore(raw_scores)

    def save(self, path: str | Path) -> None:
        payload = {
            "local_bank": self.local_bank,
            "global_mean": self.global_mean,
            "global_precision": self.global_precision,
            "color_mean": self.color_mean,
            "color_precision": self.color_precision,
            "geometry_mean": self.geometry_mean,
            "geometry_precision": self.geometry_precision,
            "calibration": (
                self.calibration.__dict__
                if self.calibration is not None
                else None
            ),
            "threshold": self.threshold,
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        config: AOIConfig,
    ) -> "NormalReference":
        payload = torch.load(
            path, map_location="cpu", weights_only=False
        )
        instance = cls(config)
        for key in [
            "local_bank",
            "global_mean",
            "global_precision",
            "color_mean",
            "color_precision",
            "geometry_mean",
            "geometry_precision",
            "threshold",
        ]:
            setattr(instance, key, payload.get(key))
        calibration = payload.get("calibration")
        instance.calibration = (
            ScoreCalibration(**calibration)
            if isinstance(calibration, dict)
            else None
        )
        instance._clear_local_bank_cache()
        return instance
