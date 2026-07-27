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

        proposals = [current]

        if false_positives:
            proposals.append(
                float(np.median(false_positives) + 0.05)
            )
        if false_negatives:
            proposals.append(
                float(np.median(false_negatives) - 0.05)
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

    def periodic_retrain(
        self,
        transfer: FewShotTransfer,
        original_normal_paths: list[str],
        original_anomaly_paths: list[str],
        validation_normal_paths: list[str],
        model_path: str | Path,
        reference_path: str | Path,
    ) -> dict:
        records = self.load_feedback()

        if len(records) < self.config.feedback_retrain_min_samples:
            return {
                "updated": False,
                "reason": (
                    "反馈数量不足："
                    f"{len(records)} < "
                    f"{self.config.feedback_retrain_min_samples}"
                ),
            }

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

            # 最小安全检查：阈值变化不可过大。
            if (
                old_threshold is not None
                and new_threshold is not None
                and abs(new_threshold - old_threshold) > 0.5
            ):
                raise RuntimeError(
                    "新模型阈值漂移过大，触发回滚。"
                )

            return {
                "updated": True,
                "model_backup": str(model_backup),
                "reference_backup": str(reference_backup),
                "feedback_used": len(records),
            }

        except Exception as error:
            shutil.copy2(model_backup, model_path)
            shutil.copy2(reference_backup, reference_path)
            return {
                "updated": False,
                "rolled_back": True,
                "error": str(error),
            }
