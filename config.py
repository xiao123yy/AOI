from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch


@dataclass
class AOIConfig:
    # Paths. Relative paths are resolved against config.json.
    root: str = "."
    student_checkpoint: str = (
        "model/convnextv2_tiny_22k_384_ema.pt"
    )
    teacher_checkpoint: str = (
        "model/convnext_large_22k_1k_224.pth"
    )
    workspace: str = "aoi_full_workspace"
    industrial_checkpoint: str = (
        "aoi_full_workspace/checkpoints/industrial_pretrained.pth"
    )

    # Four public datasets.
    mvtec_ad_root: str = "data/mvtec_ad"
    mvtec_loco_root: str = "data/mvtec_loco_ad"
    visa_root: str = "data/visa"
    dagm_root: str = "data/dagm2007"

    # Runtime.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    # Image resolution.
    train_size: int = 384
    global_size: int = 384
    roi_size: int = 384
    topk_rois: int = 2
    local_top_ratio: float = 0.01

    # Backbone implementation. Use c2_hard_b for the experimental
    # cluster-centric FFN fast mode (128/32/8 centroids in stages 1-3).
    backbone_mode: str = "dense"

    # False = 完全关闭正常参考库：推理只用监督分数，
    # 阈值在 adapt 时按监督分单独校准。
    enable_reference: bool = True
    enable_memory_local: bool = True
    enable_roi_refinement: bool = False
    local_bank_chunk_size: int = 4096

    # Model heads.
    component_slots: int = 8
    geometry_dims: int = 6
    threshold_quantile: float = 0.95

    # Normal reference.
    max_local_tokens: int = 12000
    local_tokens_per_image: int = 128

    # Public industrial training.
    public_batch_size: int = 8
    public_num_workers: int = 4
    public_epochs: int = 5
    public_steps_per_epoch: int = 300
    public_lr_head: float = 2e-4
    public_lr_backbone: float = 1e-5
    public_weight_decay: float = 1e-4
    public_global_weight: float = 0.5
    public_local_weight: float = 0.5
    public_segmentation_weight: float = 0.7
    public_component_weight: float = 0.2
    public_geometry_weight: float = 0.05
    public_rank_weight: float = 0.2
    # 尺寸/尺度异常：显式 scale 回归头权重 + 公共正常图干预概率。
    public_scale_weight: float = 0.3
    scale_intervention_probability: float = 0.12

    # Target 100-normal + 30-anomaly transfer.
    batch_size: int = 8
    num_workers: int = 4
    head_epochs: int = 3
    stage4_epochs: int = 3
    stage3_epochs: int = 3
    real_correction_epochs: int = 2
    lr_head: float = 1e-4
    lr_stage4: float = 1e-5
    lr_stage3: float = 3e-6
    weight_decay: float = 1e-4

    # Score fusion.
    supervised_weight: float = 1.0
    memory_local_weight: float = 0.35
    memory_global_weight: float = 0.35
    color_weight: float = 0.15
    geometry_weight: float = 0.15

    # Feedback.
    feedback_retrain_min_samples: int = 20
    feedback_max_memory_add: int = 64
    rollback_tolerance: float = 0.01

    target_normal_fpr: float = 0.05

    # 阈值安全 margin 系数（2.1 阈值校准稳健化）。
    # 0.0 = 旧行为（阈值 = conformal 分位点本身）；
    # >0 时 threshold = 分位点 + margin_scale * max(0.05, 0.1*std(校准分))。
    threshold_margin: float = 0.0

    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def student_checkpoint_path(self) -> Path:
        return Path(self.student_checkpoint)

    @property
    def teacher_checkpoint_path(self) -> Path:
        return Path(self.teacher_checkpoint)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace)

    @property
    def industrial_checkpoint_path(self) -> Path:
        return Path(self.industrial_checkpoint)

    @property
    def split_root(self) -> Path:
        return self.workspace_path / "splits"

    def ensure_dirs(self) -> None:
        for relative in [
            "checkpoints",
            "splits",
            "normal_reference",
            "target_adaptation",
            "evaluation",
            "deployment",
            "feedback",
            "logs",
        ]:
            (self.workspace_path / relative).mkdir(
                parents=True,
                exist_ok=True,
            )

    def validate_paths(self) -> None:
        if self.backbone_mode != "dense":
            # C²-FFN 实验已证不可用（目标域 AUROC 0.446 vs dense 0.965，
            # 端到端延迟无收益），在配置层直接拦截，避免误用旧配置。
            raise ValueError(
                "backbone_mode 仅支持 'dense'："
                f"收到 {self.backbone_mode!r}。C²-FFN 实验性骨干"
                "已验证不可用，见《项目详解》12.2 / 《优化计划》C2。"
            )

        if not self.student_checkpoint_path.exists():
            raise FileNotFoundError(
                "Student checkpoint does not exist: "
                f"{self.student_checkpoint_path}"
            )

        if (
            self.teacher_checkpoint
            and not self.teacher_checkpoint_path.exists()
        ):
            print(
                "[config] teacher checkpoint not found; "
                "teacher distillation is unavailable:",
                self.teacher_checkpoint_path,
            )

        dataset_paths = {
            "MVTec AD": Path(self.mvtec_ad_root),
            "MVTec LOCO AD": Path(self.mvtec_loco_root),
            "VisA": Path(self.visa_root),
            "DAGM": Path(self.dagm_root),
        }
        for name, path in dataset_paths.items():
            if not path.exists():
                print(f"[config] {name} directory not found: {path}")

        if not self.industrial_checkpoint_path.exists():
            print(
                "[config] industrial checkpoint has not been trained yet:",
                self.industrial_checkpoint_path,
            )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _resolve_path(value: str, base_dir: Path) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())

    @classmethod
    def load(cls, path: str | Path) -> "AOIConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        # utf-8-sig: tolerates a leading BOM (Windows editors, e.g. Notepad,
        # save UTF-8 configs with BOM by default) while accepting BOM-less files.
        data = json.loads(
            config_path.read_text(encoding="utf-8-sig")
        )
        base_dir = config_path.parent

        for key in [
            "root",
            "student_checkpoint",
            "teacher_checkpoint",
            "workspace",
            "industrial_checkpoint",
            "mvtec_ad_root",
            "mvtec_loco_root",
            "visa_root",
            "dagm_root",
        ]:
            if key in data and data[key]:
                data[key] = cls._resolve_path(
                    str(data[key]),
                    base_dir,
                )

        config = cls(**data)
        config.ensure_dirs()
        config.validate_paths()

        print("[config] base_dir:", base_dir)
        print("[config] student:", config.student_checkpoint_path)
        print("[config] workspace:", config.workspace_path)
        print(
            "[config] industrial checkpoint:",
            config.industrial_checkpoint_path,
        )
        return config
