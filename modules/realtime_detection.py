from __future__ import annotations

from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import AOIConfig
from aoi_model import AOIMultiBranchModel, TemporalLogicHead
from modules.normal_reference import NormalReference
from utils.image import (
    geometry_statistics_array,
    lab_statistics_array,
    load_rgb_array,
    nms_boxes,
    normalize_array,
    normalize_image,
)


class AOIRealtimeDetector:
    """
    Problem 1: real-time image/video anomaly detection.

    Image:
    one global forward -> top-k original-resolution ROIs -> score fusion.

    Video:
    frame detector -> optional GRU temporal logic head.
    """

    def __init__(
        self,
        config: AOIConfig,
        model: AOIMultiBranchModel,
        reference: NormalReference,
        temporal_head: TemporalLogicHead | None = None,
    ):
        self.config = config
        self.model = model.to(config.device).eval()
        self.reference = reference
        self.temporal_head = (
            temporal_head.to(config.device).eval()
            if temporal_head is not None
            else None
        )

    def _synchronize(self) -> None:
        if self.config.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.inference_mode()
    def _forward_pil(
        self,
        image: Image.Image | np.ndarray,
        size: int,
    ) -> dict:
        if isinstance(image, np.ndarray):
            tensor = normalize_array(image, size)[None].to(
                self.config.device
            )
        else:
            tensor = normalize_image(image, size)[None].to(
                self.config.device
            )
        self._synchronize()
        start = time.perf_counter()
        output = self.model(tensor)
        self._synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "output": output,
            "elapsed_ms": elapsed_ms,
        }

    def _select_rois(
        self,
        image: Image.Image | np.ndarray,
        heatmap: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        heatmap_height, heatmap_width = heatmap.shape
        if isinstance(image, np.ndarray):
            original_height, original_width = image.shape[:2]
        else:
            original_width, original_height = image.size
        candidate_count = min(
            self.config.topk_rois * 8,
            heatmap_height * heatmap_width,
        )
        if candidate_count <= 0:
            return []

        flat = heatmap.reshape(-1)
        indices = np.argpartition(flat, -candidate_count)[
            -candidate_count:
        ]
        indices = indices[np.argsort(flat[indices])[::-1]]

        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        roi_size = min(
            self.config.roi_size,
            original_width,
            original_height,
        )

        for index in indices:
            y = int(index // heatmap_width)
            x = int(index % heatmap_width)
            center_x = int(
                (x + 0.5) / heatmap_width * original_width
            )
            center_y = int(
                (y + 0.5) / heatmap_height * original_height
            )
            x0 = int(
                np.clip(
                    center_x - roi_size // 2,
                    0,
                    max(0, original_width - roi_size),
                )
            )
            y0 = int(
                np.clip(
                    center_y - roi_size // 2,
                    0,
                    max(0, original_height - roi_size),
                )
            )
            boxes.append((x0, y0, x0 + roi_size, y0 + roi_size))
            scores.append(float(heatmap[y, x]))

        kept = nms_boxes(boxes, scores)
        return [
            boxes[index]
            for index in kept[: self.config.topk_rois]
        ]

    @torch.inference_mode()
    def inspect_image(
        self,
        image_path: str | Path,
    ) -> dict:
        # 一次 cv2 解码：骨干前向 / LAB / 几何统计共享同一数组，
        # 避免同一张大图被全图转换 3 次 + PIL<->numpy 往返。
        image = load_rgb_array(image_path)

        self._synchronize()
        total_start = time.perf_counter()

        # =====================================================
        # 1. 唯一一次全图骨干网络前向
        # =====================================================
        global_result = self._forward_pil(
            image,
            self.config.global_size,
        )

        global_output = global_result["output"]
        global_forward_ms = float(
            global_result["elapsed_ms"]
        )

        supervised_global = float(
            global_output[
                "final_logit"
            ][0]
            .float()
            .cpu()
        )

        # =====================================================
        # 2. 正常参考评分（仅 enable_reference 时启用）
        # 直接复用本次前向中的F16和F32，不再调用reference.score_image()；
        # 参考库关闭时完全不提取特征，CPU 颜色/几何统计整段跳过。
        # =====================================================
        reference_start = time.perf_counter()

        if not self.config.enable_reference:
            # 完全关闭正常参考库：融合时仅使用监督分数。
            reference_scores = {
                "memory_local": 0.0,
                "memory_global": 0.0,
                "color": 0.0,
                "geometry": 0.0,
            }
        else:
            f16_feature = (
                global_output["features"]["f16"][0]
            )

            local_tokens = (
                f16_feature
                .permute(1, 2, 0)
                .reshape(
                    -1,
                    f16_feature.shape[0],
                )
                .contiguous()
            )

            global_feature = (
                global_output[
                    "features"
                ]["f32"]
                .mean(dim=(-2, -1))[0]
                .float()
                .cpu()
                .numpy()
            )

            color_feature = lab_statistics_array(image)
            geometry_feature = geometry_statistics_array(
                image,
                image.shape[1],
                image.shape[0],
            )

            if self.config.enable_memory_local:
                # GPU运算开始前同步，保证耗时统计准确。
                self._synchronize()

                reference_scores = (
                    self.reference
                    .score_reused_features(
                        local_tokens=local_tokens,
                        global_feature=global_feature,
                        color=color_feature,
                        geometry=geometry_feature,
                    )
                )

                self._synchronize()
            else:
                reference_scores = (
                    self.reference.score_fast_features(
                        global_feature=global_feature,
                        color=color_feature,
                        geometry=geometry_feature,
                    )
                )

        reference_ms = (
            time.perf_counter()
            - reference_start
        ) * 1000.0

        # =====================================================
        # 3. ROI精修是可选分支
        # 第一轮关闭，防止额外一次骨干网络前向
        # =====================================================
        boxes: list[
            tuple[int, int, int, int]
        ] = []
        roi_scores: list[float] = []
        roi_forward_ms = 0.0

        if (
            self.config.enable_roi_refinement
            and self.config.topk_rois > 0
        ):
            # ROI 精修关闭时无需把热力图转成 numpy，这里按需生成。
            supervised_heatmap = (
                torch.sigmoid(
                    global_output[
                        "local_logits"
                    ][0, 0]
                )
                .float()
                .cpu()
                .numpy()
            )

            boxes = self._select_rois(
                image,
                supervised_heatmap,
            )

            if boxes:
                roi_batch = torch.stack(
                    [
                        normalize_array(
                            image[y0 : y1, x0 : x1],
                            self.config.roi_size,
                        )
                        for (x0, y0, x1, y1) in boxes
                    ]
                ).to(self.config.device)

                self._synchronize()
                roi_start = time.perf_counter()

                roi_output = self.model(
                    roi_batch
                )

                self._synchronize()
                roi_forward_ms = (
                    time.perf_counter()
                    - roi_start
                ) * 1000.0

                roi_scores = (
                    roi_output[
                        "final_logit"
                    ]
                    .float()
                    .cpu()
                    .numpy()
                    .tolist()
                )

        strongest_roi = (
            max(roi_scores)
            if roi_scores
            else supervised_global
        )

        supervised_score = max(
            supervised_global,
            strongest_roi,
        )

        # =====================================================
        # 4. 最终融合
        # =====================================================
        fused_score = (
            self.config.supervised_weight
            * supervised_score

            + self.config.memory_local_weight
            * reference_scores[
                "memory_local"
            ]

            + self.config.memory_global_weight
            * reference_scores[
                "memory_global"
            ]

            + self.config.color_weight
            * reference_scores["color"]

            + self.config.geometry_weight
            * reference_scores[
                "geometry"
            ]
        )

        threshold = float(
            self.reference.threshold
            if self.reference.threshold
            is not None
            else 0.0
        )

        self._synchronize()
        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        other_ms = max(
            0.0,
            total_ms
            - global_forward_ms
            - reference_ms
            - roi_forward_ms,
        )

        branch_scores = {
            "supervised_global": (
                supervised_global
            ),
            "strongest_roi": strongest_roi,
            "memory_local": float(
                reference_scores[
                    "memory_local"
                ]
            ),
            "memory_global": float(
                reference_scores[
                    "memory_global"
                ]
            ),
            "color": float(
                reference_scores["color"]
            ),
            "geometry": float(
                reference_scores[
                    "geometry"
                ]
            ),
        }

        if not self.config.enable_reference:
            # 参考库关闭时四个分支恒为 0.0，max 会误选它们；
            # 此时主导分支恒为监督分。
            dominant_branch = "supervised_global"
        else:
            dominant_branch = max(
                branch_scores,
                key=lambda key: branch_scores[key],
            )

        return {
            "image": str(image_path),
            "is_anomaly": bool(
                fused_score >= threshold
            ),
            "score": float(fused_score),
            "threshold": threshold,
            "dominant_branch": (
                dominant_branch
            ),
            "branch_scores": branch_scores,
            "roi_boxes": boxes,
            "roi_scores": roi_scores,
            "latency_ms": float(total_ms),

            # 用来精确判断剩余瓶颈。
            "latency_breakdown_ms": {
                "global_forward": (
                    global_forward_ms
                ),
                "normal_reference": float(
                    reference_ms
                ),
                "roi_forward": float(
                    roi_forward_ms
                ),
                "other": float(other_ms),
            },
        }

    @staticmethod
    def _safe_auc(
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> float:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    def evaluate_dataset(
        self,
        normal_paths: list[str],
        seen_anomaly_paths: list[str] | None = None,
        unseen_anomaly_paths: list[str] | None = None,
        output_dir: str | Path | None = None,
        threshold_sweep: bool = False,
    ) -> dict:
        seen_anomaly_paths = seen_anomaly_paths or []
        unseen_anomaly_paths = unseen_anomaly_paths or []
        items = [
            *[(path, 0, "normal") for path in normal_paths],
            *[(path, 1, "seen") for path in seen_anomaly_paths],
            *[(path, 1, "unseen") for path in unseen_anomaly_paths],
        ]
        if not items:
            raise ValueError("Evaluation dataset is empty.")

        # Warm-up is excluded from metrics.
        self.inspect_image(items[0][0])

        rows: list[dict] = []
        for index, (path, label, group) in enumerate(items, start=1):
            result = self.inspect_image(path)
            rows.append({
                "path": path,
                "label": label,
                "group": group,
                "prediction": int(result["is_anomaly"]),
                "score": float(result["score"]),
                "threshold": float(result["threshold"]),
                "latency_ms": float(result["latency_ms"]),
                "dominant_branch": result["dominant_branch"],
            })
            if index % 20 == 0 or index == len(items):
                print(f"[evaluate] {index}/{len(items)}")

        labels = np.asarray([row["label"] for row in rows], dtype=int)
        predictions = np.asarray(
            [row["prediction"] for row in rows],
            dtype=int,
        )
        scores = np.asarray([row["score"] for row in rows], dtype=float)
        groups = np.asarray([row["group"] for row in rows])
        latencies = np.asarray(
            [row["latency_ms"] for row in rows],
            dtype=float,
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        normal_mask = groups == "normal"
        seen_mask = groups == "seen"
        unseen_mask = groups == "unseen"

        metrics = {
            "count": int(len(rows)),
            # C4：记录评估时部署的阈值，供 check-deploy 核对一致性。
            "deploy_threshold": (
                float(self.reference.threshold)
                if self.reference.threshold is not None
                else None
            ),
            "overall_auroc": self._safe_auc(labels, scores),
            "seen_auroc": self._safe_auc(
                labels[normal_mask | seen_mask],
                scores[normal_mask | seen_mask],
            ),
            "unseen_auroc": self._safe_auc(
                labels[normal_mask | unseen_mask],
                scores[normal_mask | unseen_mask],
            ),
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "normal_fpr": float(fp / max(1, fp + tn)),
            "seen_recall": (
                float(predictions[seen_mask].mean())
                if seen_mask.any()
                else float("nan")
            ),
            "unseen_recall": (
                float(predictions[unseen_mask].mean())
                if unseen_mask.any()
                else float("nan")
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            # 3.1：正常分数尺度（std），供 feedback-retrain 的
            # 尺度感知阈值漂移守卫使用。
            "normal_score_std": (
                float(scores[normal_mask].std())
                if normal_mask.any()
                else float("nan")
            ),
            "mean_latency_ms": float(latencies.mean()),
            "p95_latency_ms": float(np.quantile(latencies, 0.95)),
        }

        if threshold_sweep:
            # C1：区分"模型上限"与"部署点"。
            # 候选阈值 = 全体分数 1%~99% 分位（99 点）∪ 当前部署阈值。
            candidate_thresholds = np.unique(
                np.quantile(scores, np.linspace(0.01, 0.99, 99))
            )
            if self.reference.threshold is not None:
                candidate_thresholds = np.unique(
                    np.concatenate(
                        [
                            candidate_thresholds,
                            [
                                float(
                                    self.reference.threshold
                                )
                            ],
                        ]
                    )
                )

            best_point: dict | None = None
            best_f1 = -1.0
            for candidate in candidate_thresholds:
                candidate_predictions = (
                    scores >= candidate
                ).astype(int)
                candidate_f1 = float(
                    f1_score(
                        labels,
                        candidate_predictions,
                        zero_division=0,
                    )
                )
                if candidate_f1 > best_f1:
                    best_f1 = candidate_f1
                    best_point = {
                        "threshold": float(candidate),
                        "precision": float(
                            precision_score(
                                labels,
                                candidate_predictions,
                                zero_division=0,
                            )
                        ),
                        "recall": float(
                            recall_score(
                                labels,
                                candidate_predictions,
                                zero_division=0,
                            )
                        ),
                        "f1": candidate_f1,
                    }
            if best_point is not None:
                metrics["threshold_sweep"] = {
                    "best_f1_point": best_point,
                }

            # P@FPR=target：阈值取正常分的 (1-target_fpr) 分位。
            normal_scores = scores[normal_mask]
            if len(normal_scores) > 0:
                target_fpr = float(
                    self.config.target_normal_fpr
                )
                sweep_threshold = float(
                    np.quantile(
                        normal_scores,
                        1.0 - target_fpr,
                    )
                )
                sweep_predictions = (
                    scores >= sweep_threshold
                ).astype(int)
                metrics.setdefault(
                    "threshold_sweep",
                    {},
                )[
                    f"precision_at_fpr_{target_fpr}"
                ] = {
                    "threshold": sweep_threshold,
                    "fpr_actual": float(
                        (
                            normal_scores >= sweep_threshold
                        ).mean()
                    ),
                    "precision": float(
                        precision_score(
                            labels,
                            sweep_predictions,
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            labels,
                            sweep_predictions,
                            zero_division=0,
                        )
                    ),
                }

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with (output_path / "predictions.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            failed = [
                row for row in rows
                if row["label"] != row["prediction"]
            ]
            with (output_path / "failed_cases.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(failed)

            (output_path / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

    @torch.inference_mode()
    def inspect_video(
        self,
        video_path: str | Path,
        frame_stride: int = 5,
        max_frames: int = 200,
    ) -> dict:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)

        frame_results = []
        frame_features = []
        frame_index = 0

        while len(frame_results) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = normalize_image(
                image,
                self.config.global_size,
            )[None].to(self.config.device)
            output = self.model(tensor)

            local_score = float(
                torch.sigmoid(output["local_logits"]).amax().cpu()
            )
            global_score = float(
                output["final_logit"][0].float().cpu()
            )
            component = (
                torch.sigmoid(output["component_logits"][0])
                .float()
                .cpu()
                .numpy()
            )
            geometry = (
                output["geometry"][0].float().cpu().numpy()
            )
            feature = np.concatenate(
                [
                    np.array([global_score, local_score]),
                    component[:3],
                    geometry[:3],
                ]
            ).astype(np.float32)

            frame_features.append(feature)
            frame_results.append({
                "frame_index": frame_index,
                "global_score": global_score,
                "local_score": local_score,
            })
            frame_index += 1

        capture.release()

        temporal_score = 0.0
        states: list[int] = []
        if self.temporal_head is not None and frame_features:
            sequence = torch.from_numpy(
                np.stack(frame_features)
            )[None].to(self.config.device)
            temporal_output = self.temporal_head(sequence)
            temporal_score = float(
                temporal_output["sequence_logit"][0].float().cpu()
            )
            states = (
                temporal_output["state_logits"][0]
                .argmax(dim=-1)
                .cpu()
                .numpy()
                .tolist()
            )

        frame_max = max(
            [item["global_score"] for item in frame_results],
            default=0.0,
        )
        final_score = max(frame_max, temporal_score)
        threshold = float(
            self.reference.threshold
            if self.reference.threshold is not None
            else 0.0
        )
        return {
            "video": str(video_path),
            "frames_processed": len(frame_results),
            "frame_results": frame_results,
            "state_sequence": states,
            "temporal_score": temporal_score,
            "final_score": float(final_score),
            "threshold": threshold,
            "is_anomaly": bool(final_score >= threshold),
        }