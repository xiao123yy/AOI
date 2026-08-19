from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import shutil

import numpy as np
import torch

from config import AOIConfig
from aoi_model import AOIMultiBranchModel
from modules.fewshot_transfer import FewShotTransfer
from modules.normal_reference import NormalReference
from modules.realtime_detection import AOIRealtimeDetector


@dataclass
class FeedbackRecord:
    image_path: str
    predicted_label: int
    corrected_label: int
    predicted_score: float
    defect_type: str = ""
    note: str = ""
    timestamp: str = ""


class FeedbackOptimizer:
    """
    问题3：用户反馈驱动优化。

    即时更新：
    - 保存误报/漏报；
    - 正常反馈可加入候选正常库；
    - 调整阈值建议。

    周期更新：
    - 只微调任务头/Stage4；
    - 使用原始100/30样本重放；
    - 在验证集上检查；
    - 性能下降则回滚。
    """

    def __init__(
        self,
        config: AOIConfig,
        model: AOIMultiBranchModel,
        reference: NormalReference,
    ):
        self.config = config
        self.model = model
        self.reference = reference

        self.feedback_dir = (
            config.workspace_path / "feedback"
        )
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.jsonl"

    def add_feedback(
        self,
        image_path: str,
        predicted_label: int,
        corrected_label: int,
        predicted_score: float,
        defect_type: str = "",
        note: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            image_path=image_path,
            predicted_label=int(predicted_label),
            corrected_label=int(corrected_label),
            predicted_score=float(predicted_score),
            defect_type=defect_type,
            note=note,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        with self.feedback_file.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                )
                + "\n"
            )

        return record

    def load_feedback(self) -> list[FeedbackRecord]:
        if not self.feedback_file.exists():
            return []

        records = []
        for line in self.feedback_file.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                records.append(
                    FeedbackRecord(**json.loads(line))
                )
        return records

    def threshold_suggestion(self) -> float | None:
        records = self.load_feedback()
        false_positives = [
            item.predicted_score
            for item in records
            if item.predicted_label == 1
            and item.corrected_label == 0
        ]
        false_negatives = [
            item.predicted_score
            for item in records
            if item.predicted_label == 0
            and item.corrected_label == 1
        ]

        current = self.reference.threshold
        if current is None:
            return None

        # 3.2 自适应步长：步长 = 0.2 * std(所有反馈分数)，
        # 样本不足时回退到固定 0.05，避免绝对步长不随分数尺度自适应。
        all_scores = [
            item.predicted_score for item in records
        ]
        step = 0.05
        if len(all_scores) >= 2:
            score_std = float(np.std(all_scores))
            if score_std > 0.0:
                step = 0.2 * score_std

        proposals = [current]

        if false_positives:
            proposals.append(
                float(np.median(false_positives) + step)
            )
        if false_negatives:
            proposals.append(
                float(np.median(false_negatives) - step)
            )

        return float(np.median(proposals))

    def apply_immediate_update(self) -> dict:
        suggestion = self.threshold_suggestion()
        old_threshold = self.reference.threshold

        if suggestion is not None:
            # 限制单次调整，避免操作员少量反馈使阈值剧烈变化。
            maximum_change = 0.1
            self.reference.threshold = float(
                np.clip(
                    suggestion,
                    old_threshold - maximum_change,
                    old_threshold + maximum_change,
                )
            )

        return {
            "old_threshold": old_threshold,
            "new_threshold": self.reference.threshold,
            "feedback_count": len(self.load_feedback()),
        }

    def _validation_metrics(
        self,
        model: AOIMultiBranchModel,
        reference: NormalReference,
        normal_paths: list[str],
        anomaly_paths: list[str],
    ) -> dict:
        """用部署同款推理流程在验证集上计算 FPR / AUROC。"""
        detector = AOIRealtimeDetector(
            config=self.config,
            model=model,
            reference=reference,
        )
        return detector.evaluate_dataset(
            normal_paths=normal_paths,
            seen_anomaly_paths=anomaly_paths,
            unseen_anomaly_paths=[],
            output_dir=None,
        )

    def periodic_retrain(
        self,
        transfer: FewShotTransfer,
        original_normal_paths: list[str],
        original_anomaly_paths: list[str],
        validation_normal_paths: list[str],
        model_path: str | Path,
        reference_path: str | Path,
        validation_anomaly_paths: list[str] | None = None,
    ) -> dict:
        records = self.load_feedback()
        validation_anomaly_paths = validation_anomaly_paths or []

        if len(records) < self.config.feedback_retrain_min_samples:
            return {
                "updated": False,
                "reason": (
                    "反馈数量不足："
                    f"{len(records)} < "
                    f"{self.config.feedback_retrain_min_samples}"
                ),
            }

        # 3.1 重训前基线：旧模型在验证集上的 FPR / AUROC，
        # 用于重训后的"显著下降"对比。
        baseline_metrics = None
        if validation_normal_paths:
            baseline_metrics = self._validation_metrics(
                self.model,
                self.reference,
                validation_normal_paths,
                validation_anomaly_paths,
            )

        corrected_normal = [
            item.image_path
            for item in records
            if item.corrected_label == 0
        ]
        corrected_anomaly = [
            item.image_path
            for item in records
            if item.corrected_label == 1
        ]

        train_normal = list(
            dict.fromkeys(
                original_normal_paths + corrected_normal
            )
        )
        train_anomaly = list(
            dict.fromkeys(
                original_anomaly_paths + corrected_anomaly
            )
        )

        backup_dir = self.feedback_dir / "rollback"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_backup = backup_dir / f"model_{timestamp}.pth"
        reference_backup = backup_dir / f"reference_{timestamp}.pth"

        shutil.copy2(model_path, model_backup)
        shutil.copy2(reference_path, reference_backup)

        old_threshold = self.reference.threshold

        try:
            transfer.adapt(
                normal_paths=train_normal,
                anomaly_paths=train_anomaly,
                output_model_path=model_path,
                output_reference_path=reference_path,
                synthetic_engine=None,
            )

            new_threshold = transfer.reference.threshold

            # 最小安全检查 1：阈值变化不可过大（宽松哨兵）。
            # 融合分数在 enable_reference=false 时是纯监督 logit，
            # 其整体偏移随重训的头部权重而变（同一数据不同 run
            # 实测漂移 0.50~1.76，验证子集 std≈0.55）。固定 0.5
            # 绝对上限会随机拦截正常更新，因此上限按旧模型在
            # 验证集上正常分数的 std 缩放：
            #   limit = max(0.5, 4.0 * 旧正常分数std)
            # 注意：logit 整体偏移本身无害（阈值随之平移、排序
            # 不变），真正的质量闸门是下面的 FPR/AUROC 验证；
            # 本哨兵只拦"阈值飞到分数空间之外"的退化重训。
            # 没有验证集时退回旧行为（0.5 绝对上限）。
            drift_limit = 0.5
            if (
                baseline_metrics is not None
                and np.isfinite(
                    baseline_metrics.get(
                        "normal_score_std",
                        float("nan"),
                    )
                )
            ):
                drift_limit = max(
                    0.5,
                    4.0
                    * float(
                        baseline_metrics["normal_score_std"]
                    ),
                )

            if (
                old_threshold is not None
                and new_threshold is not None
                and abs(new_threshold - old_threshold) > drift_limit
            ):
                raise RuntimeError(
                    "新模型阈值漂移过大，触发回滚："
                    f"{abs(new_threshold - old_threshold):.4f} "
                    f"> {drift_limit:.4f}"
                )

            # 最小安全检查 2（3.1）：新模型必须在验证集上通过
            # FPR / AUROC 校验，不满足则触发回滚。
            validation = None
            if validation_normal_paths:
                validation_metrics = self._validation_metrics(
                    transfer.model,
                    transfer.reference,
                    validation_normal_paths,
                    validation_anomaly_paths,
                )
                validation = {
                    "normal_fpr": validation_metrics["normal_fpr"],
                    "overall_auroc": validation_metrics[
                        "overall_auroc"
                    ],
                    "count": validation_metrics["count"],
                }

                fpr_limit = (
                    self.config.target_normal_fpr * 1.5
                )
                if (
                    validation_metrics["normal_fpr"]
                    > fpr_limit
                ):
                    raise RuntimeError(
                        "新模型验证集 FPR 超标，触发回滚："
                        f"{validation_metrics['normal_fpr']:.3f} "
                        f"> {fpr_limit:.3f}"
                    )

                baseline_auroc = (
                    baseline_metrics["overall_auroc"]
                    if baseline_metrics is not None
                    else float("nan")
                )
                new_auroc = validation_metrics[
                    "overall_auroc"
                ]
                if (
                    np.isfinite(baseline_auroc)
                    and np.isfinite(new_auroc)
                    and new_auroc < baseline_auroc - 0.05
                ):
                    raise RuntimeError(
                        "新模型验证集 AUROC 显著下降，触发回滚："
                        f"{new_auroc:.4f} < "
                        f"{baseline_auroc:.4f} - 0.05"
                    )

            return {
                "updated": True,
                "model_backup": str(model_backup),
                "reference_backup": str(reference_backup),
                "feedback_used": len(records),
                "validation": validation,
            }

        except Exception as error:
            shutil.copy2(model_backup, model_path)
            shutil.copy2(reference_backup, reference_path)
            self._log_rollback(
                str(error),
                model_backup,
                reference_backup,
            )
            return {
                "updated": False,
                "rolled_back": True,
                "error": str(error),
            }

    def _log_rollback(
        self,
        error: str,
        model_backup: Path,
        reference_backup: Path,
    ) -> None:
        """回滚必须留痕，避免静默失败。"""
        log_path = (
            self.feedback_dir / "rollback" / "rollback.log"
        )
        entry = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "error": error,
            "model_backup": str(model_backup),
            "reference_backup": str(reference_backup),
        }
        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )
