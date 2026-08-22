from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import AOIConfig
from aoi_model import AOIMultiBranchModel, TemporalLogicHead
from modules.feedback_optimization import FeedbackOptimizer
from modules.fewshot_transfer import FewShotTransfer
from modules.normal_reference import NormalReference
from modules.missing_fewer_e7 import MissingFewerReference
from modules.realtime_detection import AOIRealtimeDetector
from modules.synthetic_engine import SyntheticEngine
from utils.paths import (
    build_public_and_target_split,
    list_images,
    load_jsonl,
)


def build_model(
    config: AOIConfig,
    full_checkpoint: str | Path | None = None,
) -> AOIMultiBranchModel:
    model = AOIMultiBranchModel(
        student_checkpoint=config.student_checkpoint,
        component_slots=config.component_slots,
        geometry_dims=config.geometry_dims,
        local_top_ratio=config.local_top_ratio,
        backbone_mode=config.backbone_mode,
        enable_missing_fewer=config.missing_fewer_enabled,
    )

    if full_checkpoint is None:
        print("[model] ImageNet initialization:", config.student_checkpoint_path)
        return model

    checkpoint_path = Path(full_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = (
        payload["model"]
        if isinstance(payload, dict)
        and "model" in payload
        and isinstance(payload["model"], dict)
        else payload
    )
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )
    print("[model] loaded:", checkpoint_path)
    print(
        f"[model] missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )
    if missing:
        print("[model] missing keys:", missing[:10])
    if unexpected:
        print("[model] unexpected keys:", unexpected[:10])
    return model


def load_deployed(
    config: AOIConfig,
) -> tuple[AOIMultiBranchModel, NormalReference]:
    model_path = config.workspace_path / "deployment" / "target_model.pth"
    reference_path = (
        config.workspace_path
        / "deployment"
        / "normal_reference.pth"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Target model not found: {model_path}")
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Normal reference not found: {reference_path}"
        )

    model = build_model(config)

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        "[model] target checkpoint loaded:",
        model_path,
    )
    print(
        "[model] target missing/unexpected:",
        len(missing),
        len(unexpected),
    )

    model.to(config.device).eval()
    reference = NormalReference.load(reference_path, config)
    return model, reference


def load_missing_fewer_reference(config: AOIConfig) -> MissingFewerReference | None:
    path = config.workspace_path / "deployment" / "missing_fewer_e7_reference.pth"
    return MissingFewerReference.load(path, config.device) if path.exists() else None


def command_make_split(args, config: AOIConfig) -> None:
    split_dir = build_public_and_target_split(
        config=config,
        target_dataset=args.target_dataset,
        target_category=args.target_category,
        unseen_type=args.unseen_type,
        normal_budget=args.normal_budget,
        anomaly_budget=args.anomaly_budget,
        public_val_fraction=args.public_val_fraction,
        seed=args.seed,
    )
    print("[split] created:", split_dir)
    print("[split] support normal:", split_dir / "support" / "normal")
    print("[split] support anomaly:", split_dir / "support" / "anomaly")
    print("[split] query normal:", split_dir / "query" / "normal")
    print("[split] query seen:", split_dir / "query" / "seen")
    print("[split] query unseen:", split_dir / "query" / "unseen")


def command_pretrain_public(args, config: AOIConfig) -> None:
    split_dir = Path(args.split_dir)
    train_manifest = split_dir / "public_train.jsonl"
    validation_manifest = split_dir / "public_val.jsonl"

    train_records = load_jsonl(train_manifest)
    validation_records = load_jsonl(validation_manifest)
    print(
        f"[public] train={len(train_records)}, "
        f"validation={len(validation_records)}"
    )

    model = build_model(config)
    transfer = FewShotTransfer(config, model)
    if args.missing_fewer_e7:
        transfer.pretrain_missing_fewer_public(train_records, config.industrial_checkpoint_path, args.epochs, args.steps_per_epoch, args.held_out_category)
    else:
        transfer.pretrain_public(train_records=train_records, validation_records=validation_records, output_path=config.industrial_checkpoint_path, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch)


def command_adapt(args, config: AOIConfig) -> None:
    normal_paths = list_images(args.normal_dir)
    anomaly_paths = list_images(args.anomaly_dir)
    print(
        f"[adapt] normal={len(normal_paths)}, "
        f"anomaly={len(anomaly_paths)}"
    )

    if not config.industrial_checkpoint_path.exists():
        raise FileNotFoundError(
            "Industrial pretrained checkpoint does not exist: "
            f"{config.industrial_checkpoint_path}\n"
            "Run pretrain-public first."
        )

    model = build_model(
        config,
        full_checkpoint=config.industrial_checkpoint_path,
    )
    transfer = FewShotTransfer(config, model)
    deployment_dir = config.workspace_path / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    if args.missing_fewer_e7:
        # E7 target adaptation is frozen: 100N reference then 30A boundary only.
        transfer.build_zero_shot_reference(normal_paths, deployment_dir / "normal_reference.pth")
        transfer.build_missing_fewer_reference(normal_paths, deployment_dir / "missing_fewer_e7_reference.pth")
        transfer.calibrate_missing_fewer_threshold(normal_paths, anomaly_paths, args.e7_policy)
        assert transfer.missing_fewer_reference is not None
        transfer.missing_fewer_reference.save(deployment_dir / "missing_fewer_e7_reference.pth")
        torch.save(model.state_dict(), deployment_dir / "target_model.pth")
    else:
        synthetic_engine = None if args.disable_synthetic else SyntheticEngine(seed=42)
        transfer.adapt(normal_paths=normal_paths, anomaly_paths=anomaly_paths, output_model_path=deployment_dir / "target_model.pth", output_reference_path=deployment_dir / "normal_reference.pth", synthetic_engine=synthetic_engine)


def command_evaluate(args, config: AOIConfig) -> None:
    model, reference = load_deployed(config)
    detector = AOIRealtimeDetector(
        config=config,
        model=model,
        reference=reference,
        missing_fewer_reference=load_missing_fewer_reference(config),
    )
    normal_paths = list_images(args.normal_dir)
    seen_paths = (
        list_images(args.seen_dir)
        if args.seen_dir
        else []
    )
    unseen_paths = (
        list_images(args.unseen_dir)
        if args.unseen_dir
        else []
    )
    detector.evaluate_dataset(
        normal_paths=normal_paths,
        seen_anomaly_paths=seen_paths,
        unseen_anomaly_paths=unseen_paths,
        output_dir=config.workspace_path / "evaluation",
        threshold_sweep=args.threshold_sweep,
    )


def command_zero_shot_adapt(args, config: AOIConfig) -> None:
    """无样本冷启动：不训练，只用正常图建参考库 + conformal 阈值。

    对应《优化计划》2.2：交付"无样本启动"命令行封装。
    阈值口径与 adapt 完全一致：部署同款评分流程 + 有限样本
    conformal 分位点 + threshold_margin（在全部正常图上校准，
    样本量比 adapt 的验证子集更大）。
    """
    normal_paths = list_images(args.normal_dir)
    if len(normal_paths) < 5:
        raise ValueError(
            "zero-shot 至少需要 5 张正常图："
            f"实际 {len(normal_paths)} 张。"
        )
    if not config.industrial_checkpoint_path.exists():
        raise FileNotFoundError(
            "Industrial pretrained checkpoint does not exist: "
            f"{config.industrial_checkpoint_path}\n"
            "Run pretrain-public first."
        )

    model = build_model(
        config,
        full_checkpoint=config.industrial_checkpoint_path,
    )
    transfer = FewShotTransfer(config, model)

    reference = transfer.build_zero_shot_reference(normal_paths)

    # 阈值：部署同款流程对全部正常图打分 → conformal 分位点 + margin。
    target_fpr = float(config.target_normal_fpr)
    scores = np.sort(
        np.asarray(
            transfer._combined_scores(normal_paths),
            dtype=np.float64,
        )
    )
    sample_count = len(scores)
    rank = int(
        np.ceil((sample_count + 1) * (1.0 - target_fpr))
    ) - 1
    rank = min(max(rank, 0), sample_count - 1)

    margin_scale = float(
        getattr(config, "threshold_margin", 0.0)
    )
    margin = 0.0
    if margin_scale > 0.0:
        margin = margin_scale * max(
            0.05,
            0.1 * float(scores.std()),
        )
    reference.threshold = float(scores[rank] + margin)

    deployment_dir = (
        Path(args.output_dir)
        if args.output_dir
        else config.workspace_path / "deployment"
    )
    deployment_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        deployment_dir / "target_model.pth",
    )
    reference.save(deployment_dir / "normal_reference.pth")

    print(
        json.dumps(
            {
                "normal_count": sample_count,
                "target_fpr": target_fpr,
                "rank": rank,
                "margin_scale": margin_scale,
                "threshold_margin": float(margin),
                "threshold": reference.threshold,
                "model": str(
                    deployment_dir / "target_model.pth"
                ),
                "reference": str(
                    deployment_dir / "normal_reference.pth"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_check_deploy(args, config: AOIConfig) -> None:
    """C4：部署产物与最近一次评估的一致性检查。

    防止"adapt 重新生成部署后评估已过期"导致
    部署阈值与评估记录不匹配。
    """
    _, reference = load_deployed(config)
    deployed_threshold = reference.threshold

    metrics_path = (
        config.workspace_path / "evaluation" / "metrics.json"
    )
    result: dict = {
        "deployment_threshold": deployed_threshold,
        "metrics_path": str(metrics_path),
        "metrics_threshold": None,
        "consistent": None,
        "warning": None,
    }

    if not metrics_path.exists():
        result["warning"] = (
            "evaluation/metrics.json 不存在："
            "尚无评估记录，无法核对部署点。"
        )
    else:
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
        recorded = metrics.get("deploy_threshold")
        if recorded is None:
            result["warning"] = (
                "metrics.json 缺少 deploy_threshold 字段"
                "（旧版评估记录）：请重跑 evaluate 以核对当前部署点。"
            )
        else:
            result["metrics_threshold"] = recorded
            result["consistent"] = (
                deployed_threshold is not None
                and abs(
                    float(deployed_threshold) - float(recorded)
                )
                <= 1e-6
            )
            if not result["consistent"]:
                result["warning"] = (
                    "部署阈值与最近评估记录不一致："
                    "部署产物在评估之后被重新生成，评估已过期，"
                    "请重跑 evaluate。"
                )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["warning"]:
        print("[check-deploy] WARNING:", result["warning"])


def command_infer_image(args, config: AOIConfig) -> None:
    model, reference = load_deployed(config)
    detector = AOIRealtimeDetector(
        config=config,
        model=model,
        reference=reference,
        missing_fewer_reference=load_missing_fewer_reference(config),
    )
    result = detector.inspect_image(args.image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_infer_video(args, config: AOIConfig) -> None:
    model, reference = load_deployed(config)
    temporal_head = TemporalLogicHead()
    temporal_path = (
        config.workspace_path
        / "deployment"
        / "temporal_head.pth"
    )
    if temporal_path.exists():
        temporal_head.load_state_dict(
            torch.load(
                temporal_path,
                map_location="cpu",
                weights_only=True,
            )
        )
    else:
        temporal_head = None

    detector = AOIRealtimeDetector(
        config=config,
        model=model,
        reference=reference,
        temporal_head=temporal_head,
    )
    result = detector.inspect_video(
        args.video,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_feedback(args, config: AOIConfig) -> None:
    model, reference = load_deployed(config)
    manager = FeedbackOptimizer(
        config=config,
        model=model,
        reference=reference,
    )
    record = manager.add_feedback(
        image_path=args.image,
        predicted_label=args.predicted_label,
        corrected_label=args.corrected_label,
        predicted_score=args.score,
        defect_type=args.defect_type,
        note=args.note,
    )
    update = manager.apply_immediate_update()
    reference.save(
        config.workspace_path
        / "deployment"
        / "normal_reference.pth"
    )
    print(
        json.dumps(
            {
                "feedback": record.__dict__,
                "immediate_update": update,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_feedback_retrain(args, config: AOIConfig) -> None:
    model, reference = load_deployed(config)
    transfer = FewShotTransfer(config, model)
    transfer.reference = reference
    manager = FeedbackOptimizer(
        config=config,
        model=model,
        reference=reference,
    )

    validation_anomaly = (
        list_images(args.validation_anomaly_dir)
        if args.validation_anomaly_dir
        else []
    )
    result = manager.periodic_retrain(
        transfer=transfer,
        original_normal_paths=list_images(args.normal_dir),
        original_anomaly_paths=list_images(args.anomaly_dir),
        validation_normal_paths=list_images(
            args.validation_normal_dir
        ),
        validation_anomaly_paths=validation_anomaly or None,
        model_path=(
            config.workspace_path
            / "deployment"
            / "target_model.pth"
        ),
        reference_path=(
            config.workspace_path
            / "deployment"
            / "normal_reference.pth"
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Modular AOI real-time online AI inspection system"
    )
    project_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--config",
        default=str(project_root / "config.json"),
        help=(
            "Configuration file. Relative paths inside JSON are "
            "resolved relative to the JSON file."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    split = subparsers.add_parser(
        "make-split",
        help="Scan four datasets and create public/target splits",
    )
    split.add_argument("--target-dataset", default="mvtec_ad")
    split.add_argument("--target-category", default="grid")
    split.add_argument(
        "--unseen-type",
        default=None,
        help=(
            "留作 unseen 的缺陷类型；缺省/空值时不保留 unseen，"
            "全部异常进入 support（用于单缺陷类型类别）。"
        ),
    )
    split.add_argument("--normal-budget", type=int, default=100)
    split.add_argument("--anomaly-budget", type=int, default=30)
    split.add_argument(
        "--public-val-fraction",
        type=float,
        default=0.2,
    )
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(function=command_make_split)

    public = subparsers.add_parser(
        "pretrain-public",
        help="Train on MVTec AD, VisA, DAGM and MVTec LOCO AD",
    )
    public.add_argument("--split-dir", required=True)
    public.add_argument("--epochs", type=int, default=None)
    public.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
    )
    public.add_argument("--missing-fewer-e7", action="store_true", help="Run frozen-backbone E7 public core only")
    public.add_argument("--held-out-category", default="", help="LOPO target category excluded from public records")
    public.set_defaults(function=command_pretrain_public)

    adapt = subparsers.add_parser(
        "adapt",
        help="Target transfer with 100 normal and 30 anomaly images",
    )
    adapt.add_argument("--normal-dir", required=True)
    adapt.add_argument("--anomaly-dir", required=True)
    adapt.add_argument(
        "--disable-synthetic",
        action="store_true",
    )
    adapt.add_argument("--missing-fewer-e7", action="store_true", help="Frozen E7 100N reference + 30A threshold calibration")
    adapt.add_argument("--e7-policy", default="auto", choices=["auto", "f1", "balanced_accuracy", "target_fpr"])
    adapt.set_defaults(function=command_adapt)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Frozen evaluation on normal/seen/unseen query data",
    )
    evaluate.add_argument("--normal-dir", required=True)
    evaluate.add_argument("--seen-dir", default="")
    evaluate.add_argument("--unseen-dir", default="")
    evaluate.add_argument(
        "--threshold-sweep",
        action="store_true",
        help=(
            "C1：额外输出最优 F1 工作点与 P@FPR=5%% 两个口径，"
            "区分模型上限与部署点。"
        ),
    )
    evaluate.set_defaults(function=command_evaluate)

    zero_shot = subparsers.add_parser(
        "zero-shot-adapt",
        help=(
            "Zero-shot cold start: build the normal reference from "
            "normal images only, no training"
        ),
    )
    zero_shot.add_argument("--normal-dir", required=True)
    zero_shot.add_argument(
        "--output-dir",
        default="",
        help="Defaults to <workspace>/deployment",
    )
    zero_shot.set_defaults(function=command_zero_shot_adapt)

    check = subparsers.add_parser(
        "check-deploy",
        help="Check deployment artifacts against the latest evaluation",
    )
    check.set_defaults(function=command_check_deploy)

    image = subparsers.add_parser(
        "infer-image",
        help="Frozen inference for one high-resolution image",
    )
    image.add_argument("--image", required=True)
    image.set_defaults(function=command_infer_image)

    video = subparsers.add_parser(
        "infer-video",
        help="Frame anomaly detection plus temporal logic detection",
    )
    video.add_argument("--video", required=True)
    video.add_argument("--frame-stride", type=int, default=5)
    video.set_defaults(function=command_infer_video)

    feedback = subparsers.add_parser(
        "feedback",
        help="Record operator feedback and adjust immediately",
    )
    feedback.add_argument("--image", required=True)
    feedback.add_argument(
        "--predicted-label",
        required=True,
        type=int,
        choices=[0, 1],
    )
    feedback.add_argument(
        "--corrected-label",
        required=True,
        type=int,
        choices=[0, 1],
    )
    feedback.add_argument("--score", required=True, type=float)
    feedback.add_argument("--defect-type", default="")
    feedback.add_argument("--note", default="")
    feedback.set_defaults(function=command_feedback)

    retrain = subparsers.add_parser(
        "feedback-retrain",
        help="Periodic fine-tuning and rollback after feedback",
    )
    retrain.add_argument("--normal-dir", required=True)
    retrain.add_argument("--anomaly-dir", required=True)
    retrain.add_argument(
        "--validation-normal-dir",
        required=True,
    )
    retrain.add_argument(
        "--validation-anomaly-dir",
        default="",
        help="可选：验证异常图目录，提供后重训后额外校验 AUROC",
    )
    retrain.set_defaults(function=command_feedback_retrain)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    config = AOIConfig.load(args.config)
    args.function(args, config)


if __name__ == "__main__":
    main()
