from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from sklearn.metrics import roc_auc_score

from config import AOIConfig
from aoi_model import AOIMultiBranchModel
from modules.normal_reference import NormalReference
from modules.synthetic_engine import (
    SyntheticEngine,
    random_scale_pair,
    scale_intervene,
)
from utils.image import (
    geometry_statistics,
    load_rgb,
    normalize_image,
)

from modules.realtime_detection import (
    AOIRealtimeDetector,
)


def _worker_rng(dataset) -> np.random.Generator:
    """返回 dataset 的 rng，并按 DataLoader worker 重播种。

    多 worker 下各 worker 共享同一 dataset 对象（含同一初始随机状态），
    按 worker id 重播种可避免所有 worker 对同一样本做完全相同的增强。
    """
    info = torch.utils.data.get_worker_info()
    worker_id = 0 if info is None else info.id
    if worker_id != getattr(dataset, "_rng_worker_id", None):
        dataset._rng_worker_id = worker_id
        dataset.rng = np.random.default_rng(
            dataset.seed + worker_id * 1000003
        )
    return dataset.rng


class PublicIndustrialDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        config: AOIConfig,
        training: bool,
        seed: int = 42,
    ):
        self.records = records
        self.config = config
        self.training = training
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        # 几何统计按路径缓存：避免每 epoch 对全量图片重跑 Canny/轮廓。
        self._geometry_cache: dict[str, np.ndarray] = {}

        if not records:
            raise ValueError("Public industrial dataset is empty.")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_mask(
        path: str,
        size: int,
    ) -> tuple[torch.Tensor, float]:
        if not path or not Path(path).exists():
            return (
                torch.zeros(1, size, size, dtype=torch.float32),
                0.0,
            )

        with Image.open(path) as image:
            image = image.convert("L").resize(
                (size, size),
                Image.NEAREST,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array > 0.5).astype(np.float32)

        return torch.from_numpy(array)[None], 1.0

    def _cached_geometry(
        self,
        path: str,
        horizontal_flip: bool,
    ) -> np.ndarray:
        """按路径缓存几何统计（原图计算一次，跨 epoch 复用）。

        旧实现对增强后的图计算；此处改在原图上计算：
        翻转样本只修正 x 中心（宽高/面积/周长在左右翻转下不变），
        亮度/对比度 ±10% 对 Canny 边缘的影响可忽略。
        """
        cached = self._geometry_cache.get(path)
        if cached is None:
            cached = geometry_statistics(load_rgb(path))
            self._geometry_cache[path] = cached
        if horizontal_flip:
            flipped = cached.copy()
            flipped[2] = 1.0 - flipped[2] - flipped[0]
            return flipped
        return cached

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = load_rgb(record["path"])
        horizontal_flip = False
        rng = _worker_rng(self)
        label = float(record["label"])

        if self.training:
            horizontal_flip = rng.random() < 0.5
            if horizontal_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            image = ImageEnhance.Brightness(image).enhance(
                float(rng.uniform(0.9, 1.1))
            )
            image = ImageEnhance.Contrast(image).enhance(
                float(rng.uniform(0.9, 1.1))
            )

        # ── 尺度干预合成（尺寸异常任务，文档方法）──
        # 对正常图做无黑边中心缩放并给出 (sx, sy) 标签：
        #   正常样本监督 scale 目标 (0,0)；干预样本监督 (log sx, log sy)；
        #   真实异常无尺度标签（scale_valid=0，不计损失）。
        scale_log = np.zeros(2, dtype=np.float32)
        scale_valid = 1.0 if label == 0.0 else 0.0
        if (
            self.training
            and label == 0.0
            and rng.random() < self.config.scale_intervention_probability
        ):
            sx, sy = random_scale_pair(rng)
            image = scale_intervene(image, sx, sy)
            label = 1.0
            scale_valid = 1.0
            scale_log = np.array(
                [np.log(sx), np.log(sy)], dtype=np.float32
            )

        geometry = self._cached_geometry(
            str(record["path"]),
            horizontal_flip,
        )
        image_tensor = normalize_image(
            image,
            self.config.train_size,
        )
        mask, has_mask = self._load_mask(
            str(record.get("mask_path", "")),
            self.config.train_size,
        )
        if horizontal_flip:
            mask = torch.flip(mask, dims=[2])

        return {
            "image": image_tensor,
            "mask": mask,
            "has_mask": torch.tensor(has_mask, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "geometry": torch.from_numpy(geometry),
            "scale_log": torch.from_numpy(scale_log),
            "scale_valid": torch.tensor(
                scale_valid, dtype=torch.float32
            ),
            "task_type": str(record.get("task_type", "appearance")),
        }


class TargetDataset(Dataset):
    def __init__(
        self,
        normal_paths: list[str],
        anomaly_paths: list[str],
        config: AOIConfig,
        training: bool = True,
        synthetic_engine: SyntheticEngine | None = None,
        synthetic_probability: float = 0.0,
        seed: int = 42,
    ):
        self.config = config
        self.training = training
        self.synthetic_engine = synthetic_engine
        self.synthetic_probability = synthetic_probability
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.items = [
            (path, 0) for path in normal_paths
        ] + [
            (path, 1) for path in anomaly_paths
        ]

        if not self.items:
            raise ValueError("Target dataset is empty.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, label = self.items[index]
        image = load_rgb(path)
        is_synthetic = 0
        rng = _worker_rng(self)
        # 尺度标签：正常样本目标 (0,0)、尺度干预合成样本目标 (log sx, log sy)；
        # appearance/color 合成与真实异常没有尺度标签（valid=0）。
        scale_log = np.zeros(2, dtype=np.float32)
        scale_valid = 1.0 if label == 0 else 0.0

        if (
            self.training
            and label == 0
            and self.synthetic_engine is not None
            and rng.random() < self.synthetic_probability
        ):
            generator = rng.choice(
                ["appearance", "color", "geometry"]
            )
            if generator == "appearance":
                image, _ = self.synthetic_engine.appearance(image)
                scale_valid = 0.0
            elif generator == "color":
                image = self.synthetic_engine.color(image)
                scale_valid = 0.0
            else:
                image, _, sx, sy = (
                    self.synthetic_engine.scale_intervention(image)
                )
                scale_log = np.array(
                    [np.log(sx), np.log(sy)], dtype=np.float32
                )
                scale_valid = 1.0
            label = 1
            is_synthetic = 1

        return {
            "image": normalize_image(
                image,
                self.config.train_size,
            ),
            "label": torch.tensor(float(label)),
            "domain": torch.tensor(float(is_synthetic)),
            "geometry": torch.from_numpy(
                geometry_statistics(image)
            ),
            "scale_log": torch.from_numpy(scale_log),
            "scale_valid": torch.tensor(
                float(scale_valid), dtype=torch.float32
            ),
            "path": path,
        }


def _local_image_logit(
    local_logits: torch.Tensor,
    top_ratio: float,
) -> torch.Tensor:
    flat = local_logits.flatten(1)
    k = max(1, int(round(flat.shape[1] * top_ratio)))
    return torch.topk(flat, k=k, dim=1).values.mean(dim=1)


def _dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = (
        probability.sum(dim=(-2, -1))
        + target.sum(dim=(-2, -1))
    )
    return (
        1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean()


def _ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return scores.new_tensor(0.0)
    difference = positive[:, None] - negative[None, :]
    return F.relu(1.0 - difference).mean()


class FewShotTransfer:
    """
    Problem 2: few-shot/zero-shot startup and generalization.

    The same module handles two stages:
    1. public industrial training on MVTec AD, VisA, DAGM and LOCO;
    2. target transfer using 100 normal and 30 anomalous images.
    """

    def __init__(
        self,
        config: AOIConfig,
        model: AOIMultiBranchModel,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.reference = NormalReference(config)

    def _autocast(self):
        if self.config.device.startswith("cuda") and self.config.amp:
            # C²-FFN 的质心路由/聚合在 fp16 反向传播时会产生 NaN 梯度
            # （F.normalize / scatter_add / gather 的 fp16 数值不稳定），
            # GradScaler 会跳过所有含 NaN 的更新导致模型完全不学习。
            # C² 模式强制 fp32 训练。
            if self.model.backbone_mode in ("c2_hard_b", "c2_ffn_v1_b"):
                return nullcontext()
            return torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
        return nullcontext()

    def build_zero_shot_reference(
        self,
        normal_paths: Iterable[str | Path],
        output_path: str | Path | None = None,
    ) -> NormalReference:
        normal_paths = [str(path) for path in normal_paths]
        self.reference.fit(self.model, normal_paths)
        if output_path is not None:
            self.reference.save(output_path)
        return self.reference

    # ------------------------------------------------------------------
    # Public industrial training
    # ------------------------------------------------------------------
    def _stage_module(self, index: int) -> nn.Module:
        """按骨干模式取第 index 个 stage 的模块。

        dense: backbone.stages[index]；
        c2_hard_b: Stage 0-2 为 backbone.c2_stages[index].blocks，
                   Stage 3 为 backbone.stage4。
        """
        backbone = self.model.backbone
        if self.model.backbone_mode in ("c2_hard_b", "c2_ffn_v1_b"):
            if index == 3:
                return backbone.stage4
            return backbone.c2_stages[index].blocks
        return backbone.stages[index]

    def _set_public_trainable(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for module in [
            self.model.local_head,
            self.model.global_head,
            self.model.component_head,
            self.model.geometry_head,
            self.model.domain_head,
            self.model.fusion_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

        for stage_index in [2, 3]:
            for parameter in self._stage_module(
                stage_index
            ).parameters():
                parameter.requires_grad = True
            for parameter in self.model.backbone.downsample_layers[
                stage_index
            ].parameters():
                parameter.requires_grad = True

    def _public_optimizer(self) -> torch.optim.Optimizer:
        heads = []
        backbone = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone.append(parameter)
            else:
                heads.append(parameter)

        return torch.optim.AdamW(
            [
                {
                    "params": heads,
                    "lr": self.config.public_lr_head,
                },
                {
                    "params": backbone,
                    "lr": self.config.public_lr_backbone,
                },
            ],
            weight_decay=self.config.public_weight_decay,
        )

    def _public_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        images = batch["image"].to(self.config.device)
        labels = batch["label"].to(self.config.device)
        masks = batch["mask"].to(self.config.device)
        has_mask = batch["has_mask"].to(self.config.device)
        geometry_target = batch["geometry"].to(self.config.device)
        task_types = batch["task_type"]

        output = self.model(images)

        classification = F.binary_cross_entropy_with_logits(
            output["final_logit"],
            labels,
        )
        global_loss = F.binary_cross_entropy_with_logits(
            output["global_logit"],
            labels,
        )
        local_score = _local_image_logit(
            output["local_logits"],
            self.config.local_top_ratio,
        )
        local_loss = F.binary_cross_entropy_with_logits(
            local_score,
            labels,
        )

        selected = has_mask > 0.5
        if selected.any():
            target_mask = F.interpolate(
                masks[selected],
                size=output["local_logits"].shape[-2:],
                mode="nearest",
            )
            segmentation = (
                F.binary_cross_entropy_with_logits(
                    output["local_logits"][selected],
                    target_mask,
                )
                + _dice_loss(
                    output["local_logits"][selected],
                    target_mask,
                )
            )
        else:
            segmentation = classification.new_tensor(0.0)

        structure_mask = torch.tensor(
            [
                task in {"structure", "logic"}
                for task in task_types
            ],
            dtype=torch.bool,
            device=self.config.device,
        )
        if structure_mask.any():
            component_score = output["component_logits"][
                structure_mask
            ].mean(dim=1)
            component_loss = F.binary_cross_entropy_with_logits(
                component_score,
                labels[structure_mask],
            )
        else:
            component_loss = classification.new_tensor(0.0)

        geometry_loss = F.smooth_l1_loss(
            output["geometry"],
            geometry_target,
        )
        # 显式尺度回归损失：只对 normal（目标 0）与
        # 尺度干预合成图（目标 log sx, log sy）监督。
        scale_valid = (
            batch["scale_valid"].to(self.config.device).bool()
        )
        if scale_valid.any():
            scale_log_loss = F.smooth_l1_loss(
                output["scale_log"][scale_valid],
                batch["scale_log"].to(self.config.device)[
                    scale_valid
                ],
            )
        else:
            scale_log_loss = classification.new_tensor(0.0)
        ranking = _ranking_loss(
            output["final_logit"],
            labels,
        )

        loss = (
            classification
            + self.config.public_global_weight * global_loss
            + self.config.public_local_weight * local_loss
            + self.config.public_segmentation_weight * segmentation
            + self.config.public_component_weight * component_loss
            + self.config.public_geometry_weight * geometry_loss
            + self.config.public_scale_weight * scale_log_loss
            + self.config.public_rank_weight * ranking
        )

        parts = {
            "classification": float(classification.detach().cpu()),
            "global": float(global_loss.detach().cpu()),
            "local": float(local_loss.detach().cpu()),
            "segmentation": float(segmentation.detach().cpu()),
            "component": float(component_loss.detach().cpu()),
            "geometry": float(geometry_loss.detach().cpu()),
            "scale": float(scale_log_loss.detach().cpu()),
            "ranking": float(ranking.detach().cpu()),
        }
        return loss, parts

    @torch.inference_mode()
    def _validate_public(
        self,
        loader: DataLoader,
    ) -> dict[str, float]:
        self.model.eval()
        labels: list[int] = []
        scores: list[float] = []
        losses: list[float] = []

        for batch in loader:
            images = batch["image"].to(self.config.device)
            batch_labels = batch["label"].to(self.config.device)
            output = self.model(images)
            loss = F.binary_cross_entropy_with_logits(
                output["final_logit"],
                batch_labels,
            )
            labels.extend(
                batch_labels.int().cpu().numpy().tolist()
            )
            scores.extend(
                output["final_logit"].float().cpu().numpy().tolist()
            )
            losses.append(float(loss.cpu()))

        labels_array = np.asarray(labels, dtype=int)
        scores_array = np.asarray(scores, dtype=float)
        auroc = (
            float(roc_auc_score(labels_array, scores_array))
            if len(np.unique(labels_array)) > 1
            else float("nan")
        )
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "auroc": auroc,
        }

    def pretrain_public(
        self,
        train_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
        output_path: str | Path,
        epochs: int | None = None,
        steps_per_epoch: int | None = None,
    ) -> dict[str, Any]:
        train_dataset = PublicIndustrialDataset(
            train_records,
            self.config,
            training=True,
        )
        validation_dataset = PublicIndustrialDataset(
            validation_records,
            self.config,
            training=False,
        )

        label_counts: dict[int, int] = {0: 0, 1: 0}
        for record in train_records:
            label_counts[int(record["label"])] += 1
        sample_weights = [
            1.0 / max(1, label_counts[int(record["label"])])
            for record in train_records
        ]

        batch_size = min(
            self.config.public_batch_size,
            len(train_dataset),
        )
        steps = steps_per_epoch or self.config.public_steps_per_epoch
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=steps * batch_size,
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.public_num_workers,
            pin_memory=self.config.device.startswith("cuda"),
            drop_last=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.public_num_workers,
            pin_memory=self.config.device.startswith("cuda"),
        )

        self._set_public_trainable()
        optimizer = self._public_optimizer()
        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                self.config.device.startswith("cuda")
                and self.config.amp
            )
        )

        epoch_count = epochs or self.config.public_epochs
        best_auc = -np.inf
        best_state = deepcopy(self.model.state_dict())
        history: list[dict[str, float]] = []

        for epoch in range(1, epoch_count + 1):
            self.model.train()
            epoch_losses: list[float] = []

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss, _ = self._public_loss(batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(float(loss.detach().cpu()))

            validation = self._validate_public(validation_loader)
            train_loss = float(np.mean(epoch_losses))
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": validation["loss"],
                "val_auroc": validation["auroc"],
            }
            history.append(record)

            print(
                f"[public] epoch={epoch:03d} "
                f"train_loss={train_loss:.5f} "
                f"val_loss={validation['loss']:.5f} "
                f"val_auc={validation['auroc']:.5f}"
            )

            if (
                np.isfinite(validation["auroc"])
                and validation["auroc"] > best_auc
            ):
                best_auc = validation["auroc"]
                best_state = deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "best_validation_auroc": (
                    float(best_auc) if np.isfinite(best_auc) else None
                ),
                "history": history,
            },
            output_path,
        )
        output_path.with_suffix(".history.json").write_text(
            json.dumps(
                history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("[public] saved:", output_path)
        return {
            "checkpoint": str(output_path),
            "best_validation_auroc": (
                float(best_auc) if np.isfinite(best_auc) else None
            ),
        }

    # ------------------------------------------------------------------
    # Target few-shot transfer
    # ------------------------------------------------------------------
    def _set_trainable(self, stage: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for module in [
            self.model.local_head,
            self.model.global_head,
            self.model.component_head,
            self.model.geometry_head,
            self.model.domain_head,
            self.model.fusion_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True

        if stage in {"stage4", "stage3"}:
            for parameter in self._stage_module(3).parameters():
                parameter.requires_grad = True

        if stage == "stage3":
            for block in self._stage_module(2)[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

    def _optimizer(self) -> torch.optim.Optimizer:
        groups = {"head": [], "stage4": [], "stage3": []}

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            # dense: backbone.stages.N；c2: backbone.stage4 / backbone.c2_stages.N
            if name.startswith("backbone.stages.3") or name.startswith(
                "backbone.stage4"
            ):
                groups["stage4"].append(parameter)
            elif name.startswith("backbone.stages.2") or name.startswith(
                "backbone.c2_stages.2"
            ):
                groups["stage3"].append(parameter)
            else:
                groups["head"].append(parameter)

        parameter_groups = []
        for key, learning_rate in [
            ("head", self.config.lr_head),
            ("stage4", self.config.lr_stage4),
            ("stage3", self.config.lr_stage3),
        ]:
            if groups[key]:
                parameter_groups.append({
                    "params": groups[key],
                    "lr": learning_rate,
                })

        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.config.weight_decay,
        )

    @torch.inference_mode()
    def _target_auc(
        self,
        normal_paths: list[str],
        anomaly_paths: list[str],
    ) -> float:
        if not normal_paths or not anomaly_paths:
            return float("nan")

        self.model.eval()
        labels: list[int] = []
        scores: list[float] = []
        for path, label in [
            *[(path, 0) for path in normal_paths],
            *[(path, 1) for path in anomaly_paths],
        ]:
            image = load_rgb(path)
            tensor = normalize_image(
                image,
                self.config.train_size,
            )[None].to(self.config.device)
            output = self.model(tensor)
            labels.append(label)
            scores.append(
                float(output["final_logit"][0].float().cpu())
            )
        return float(roc_auc_score(labels, scores))

    def _train_phase(
        self,
        normal_paths: list[str],
        anomaly_paths: list[str],
        validation_normal: list[str],
        validation_anomaly: list[str],
        stage: str,
        epochs: int,
        synthetic_engine: SyntheticEngine | None = None,
        synthetic_probability: float = 0.0,
    ) -> None:
        self._set_trainable(stage)
        optimizer = self._optimizer()

        dataset = TargetDataset(
            normal_paths=normal_paths,
            anomaly_paths=anomaly_paths,
            config=self.config,
            training=True,
            synthetic_engine=synthetic_engine,
            synthetic_probability=synthetic_probability,
        )

        counts = {
            0: max(1, len(normal_paths)),
            1: max(1, len(anomaly_paths)),
        }
        weights = [
            1.0 / counts[label]
            for _, label in dataset.items
        ]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=max(len(dataset), self.config.batch_size * 8),
            replacement=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.device.startswith("cuda"),
        )

        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                self.config.device.startswith("cuda")
                and self.config.amp
            )
        )
        best_auc = -np.inf
        best_state = deepcopy(self.model.state_dict())

        for epoch in range(1, epochs + 1):
            self.model.train()
            losses: list[float] = []

            for batch in loader:
                images = batch["image"].to(self.config.device)
                labels = batch["label"].to(self.config.device)
                domains = batch["domain"].to(self.config.device)
                coefficient = 1.0 if synthetic_probability > 0 else 0.0

                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    output = self.model(
                        images,
                        domain_coefficient=coefficient,
                    )
                    local_score = _local_image_logit(
                        output["local_logits"],
                        self.config.local_top_ratio,
                    )
                    classification = F.binary_cross_entropy_with_logits(
                        output["final_logit"],
                        labels,
                    )
                    global_loss = F.binary_cross_entropy_with_logits(
                        output["global_logit"],
                        labels,
                    )
                    local_loss = F.binary_cross_entropy_with_logits(
                        local_score,
                        labels,
                    )
                    domain_loss = (
                        F.binary_cross_entropy_with_logits(
                            output["domain_logit"],
                            domains,
                        )
                        if synthetic_probability > 0
                        else classification.new_tensor(0.0)
                    )
                    # 结构/几何监督（E2 实验）：
                    # 目标域异常视为组件级异常，直接监督 component/geometry 头，
                    # 补上公共预训练中 structure 数据缺失导致的监督空白。
                    component_loss = (
                        F.binary_cross_entropy_with_logits(
                            output["component_logits"].mean(dim=1),
                            labels,
                        )
                    )
                    geometry_loss = F.smooth_l1_loss(
                        output["geometry"],
                        batch["geometry"].to(self.config.device),
                    )
                    # 显式尺度回归损失（适配阶段）：
                    # normal 目标 0、尺度干预合成图目标 log(sx,sy)，
                    # appearance/color 合成与真实异常无尺度标签（valid=0）。
                    scale_valid = (
                        batch["scale_valid"]
                        .to(self.config.device)
                        .bool()
                    )
                    if scale_valid.any():
                        scale_log_loss = F.smooth_l1_loss(
                            output["scale_log"][scale_valid],
                            batch["scale_log"]
                            .to(self.config.device)[scale_valid],
                        )
                    else:
                        scale_log_loss = classification.new_tensor(0.0)
                    loss = (
                        classification
                        + 0.25 * global_loss
                        + 0.25 * local_loss
                        + 0.05 * domain_loss
                        + 0.2 * component_loss
                        + 0.05 * geometry_loss
                        + self.config.public_scale_weight * scale_log_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

            validation_auc = self._target_auc(
                validation_normal,
                validation_anomaly,
            )
            print(
                f"[transfer] stage={stage} epoch={epoch:03d} "
                f"loss={np.mean(losses):.5f} "
                f"val_auc={validation_auc:.5f}"
            )
            if (
                np.isfinite(validation_auc)
                and validation_auc > best_auc
            ):
                best_auc = validation_auc
                best_state = deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)

    @torch.inference_mode()
    def _combined_scores(
        self,
        paths: list[str],
    ) -> np.ndarray:
        """
        阈值校准必须使用与最终部署完全相同的评分流程。

        不能再单独实现一套：
        supervised + normal reference。
        """
        self.model.eval()

        detector = AOIRealtimeDetector(
            config=self.config,
            model=self.model,
            reference=self.reference,
        )

        scores: list[float] = []

        for path in paths:
            result = detector.inspect_image(
                path
            )
            scores.append(
                float(result["score"])
            )

        return np.asarray(
            scores,
            dtype=np.float32,
        )

    def adapt(
        self,
        normal_paths: list[str],
        anomaly_paths: list[str],
        output_model_path: str | Path,
        output_reference_path: str | Path,
        synthetic_engine: SyntheticEngine | None = None,
    ) -> None:
        if len(normal_paths) < 20:
            raise ValueError(
                "At least 20 normal target images are required."
            )
        if len(anomaly_paths) < 2:
            raise ValueError(
                "At least 2 anomalous target images are required."
            )

        rng = np.random.default_rng(42)
        normal_array = np.asarray(normal_paths, dtype=object)
        anomaly_array = np.asarray(anomaly_paths, dtype=object)
        rng.shuffle(normal_array)
        rng.shuffle(anomaly_array)

        # 2.1 阈值校准稳健化：校准集从 20% 提升到 30%（100 张时 20→30），
        # 降低阈值对"校准集里最异常那张正常图"的敏感度。
        normal_val_count = max(8, round(len(normal_array) * 0.3))
        normal_val_count = min(normal_val_count, len(normal_array) - 8)
        anomaly_val_count = max(1, round(len(anomaly_array) * 0.2))
        anomaly_val_count = min(anomaly_val_count, len(anomaly_array) - 1)

        validation_normal = normal_array[:normal_val_count].tolist()
        train_normal = normal_array[normal_val_count:].tolist()
        validation_anomaly = anomaly_array[:anomaly_val_count].tolist()
        train_anomaly = anomaly_array[anomaly_val_count:].tolist()

        self._train_phase(
            train_normal,
            train_anomaly,
            validation_normal,
            validation_anomaly,
            stage="heads",
            epochs=self.config.head_epochs,
        )
        self._train_phase(
            train_normal,
            train_anomaly,
            validation_normal,
            validation_anomaly,
            stage="stage4",
            epochs=self.config.stage4_epochs,
        )
        self._train_phase(
            train_normal,
            train_anomaly,
            validation_normal,
            validation_anomaly,
            stage="stage3",
            epochs=self.config.stage3_epochs,
            synthetic_engine=synthetic_engine,
            synthetic_probability=(
                0.35 if synthetic_engine is not None else 0.0
            ),
        )
        self._train_phase(
            train_normal,
            train_anomaly,
            validation_normal,
            validation_anomaly,
            stage="heads",
            epochs=self.config.real_correction_epochs,
        )

        # The normal reference must be extracted from the final adapted model.
        # enable_reference=False 时完全跳过参考库，阈值按纯监督分校准。
        if self.config.enable_reference:
            self.reference.fit(self.model, train_normal)
        calibration_scores = self._combined_scores(validation_normal)
        calibration_scores = np.asarray(
            calibration_scores,
            dtype=np.float64,
        )

        calibration_scores = np.sort(
            calibration_scores
        )

        sample_count = len(calibration_scores)

        if sample_count == 0:
            raise RuntimeError(
                "阈值校准集为空，无法确定异常阈值。"
            )

        target_fpr = float(
            self.config.target_normal_fpr
        )

        if not 0.0 < target_fpr < 1.0:
            raise ValueError(
                "target_normal_fpr必须位于(0, 1)内。"
            )

        # 有限样本conformal分位点：
        # ceil((n + 1) * (1 - alpha))
        rank = int(
            np.ceil(
                (sample_count + 1)
                * (1.0 - target_fpr)
            )
        ) - 1

        rank = min(
            max(rank, 0),
            sample_count - 1,
        )

        # 2.1 阈值安全 margin：阈值 = conformal 分位点
        # + threshold_margin * max(0.05, 0.1*std(校准分))。
        # threshold_margin=0 时保持旧行为（便于 A/B 对比）。
        margin_scale = float(
            getattr(self.config, "threshold_margin", 0.0)
        )
        threshold_margin = 0.0
        if margin_scale > 0.0:
            threshold_margin = margin_scale * max(
                0.05,
                0.1 * float(calibration_scores.std()),
            )

        self.reference.threshold = float(
            calibration_scores[rank] + threshold_margin
        )

        print(
            "[transfer] calibration:",
            {
                "count": sample_count,
                "min": float(
                    calibration_scores[0]
                ),
                "median": float(
                    np.median(calibration_scores)
                ),
                "max": float(
                    calibration_scores[-1]
                ),
                "rank": rank,
                "target_fpr": target_fpr,
                "margin_scale": margin_scale,
                "threshold_margin": float(
                    threshold_margin
                ),
                "threshold": (
                    self.reference.threshold
                ),
            },
        )

        output_model_path = Path(output_model_path)
        output_reference_path = Path(output_reference_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        output_reference_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), output_model_path)
        self.reference.save(output_reference_path)

        print("[transfer] model:", output_model_path)
        print("[transfer] reference:", output_reference_path)
        print("[transfer] threshold:", self.reference.threshold)