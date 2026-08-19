"""五类统一基线实验脚本（《优化计划》2.3）。

每个目标类别在**独立 workspace** 里跑完整管线：

    make-split → pretrain-public（该类别专属 manifest，防泄漏）
    → adapt → evaluate（含 threshold sweep）

跑完把指标追加到 experiments/summary.csv，供跨类别对比。

用法：
    python scripts/run_experiment.py \
        --target-dataset mvtec_ad --target-category grid --unseen-type bent

    python scripts/run_experiment.py --all
        # 依次跑 grid / leather / capsule / transistor / dagm Class3

说明：
    - 每类用**独立进程**（--all 时顺序拉起子进程），避免显存残留；
    - --skip-pretrain 可复用该实验目录已有的 industrial_pretrained.pth；
    - 沙箱环境下请用 num_workers=0 的配置（如 config_nw0.json）。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 五个目标类别（unseen_type 为 None 表示单缺陷类型，不保留 unseen）。
ALL_CATEGORIES: list[tuple[str, str, str | None]] = [
    ("mvtec_ad", "grid", "bent"),
    ("mvtec_ad", "leather", "fold"),
    ("mvtec_ad", "capsule", "crack"),
    ("mvtec_ad", "transistor", "bent_lead"),
    ("dagm2007", "Class3", None),
]


def experiment_tag(
    target_dataset: str,
    target_category: str,
    unseen_type: str | None,
) -> str:
    tag = f"{target_dataset}_{target_category}"
    if unseen_type:
        tag = f"{tag}_unseen_{unseen_type}"
    return tag


def run_experiment(
    base_config_path: str,
    target_dataset: str,
    target_category: str,
    unseen_type: str | None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    normal_budget: int = 100,
    anomaly_budget: int = 30,
    skip_pretrain: bool = False,
    visualize: bool = False,
) -> dict:
    import torch

    from config import AOIConfig
    from modules.fewshot_transfer import FewShotTransfer
    from modules.realtime_detection import AOIRealtimeDetector
    from modules.synthetic_engine import SyntheticEngine
    from utils.paths import (
        build_public_and_target_split,
        list_images,
        load_jsonl,
    )

    import main as aoi_main

    started = time.time()
    base = AOIConfig.load(base_config_path)
    tag = experiment_tag(
        target_dataset, target_category, unseen_type
    )
    experiment_root = base.root_path / "experiments" / tag
    experiment_root.mkdir(parents=True, exist_ok=True)

    # ---- 每类独立 workspace / 独立预训练产物 ----
    config = AOIConfig.load(base_config_path)
    config.workspace = str(experiment_root)
    config.industrial_checkpoint = str(
        experiment_root
        / "checkpoints"
        / "industrial_pretrained.pth"
    )
    config.ensure_dirs()
    config.validate_paths()
    config.save(experiment_root / "config.json")

    print(f"[experiment] tag={tag}")
    print(f"[experiment] workspace={experiment_root}")

    # ---- 1. make-split ----
    split_dir = build_public_and_target_split(
        config=config,
        target_dataset=target_dataset,
        target_category=target_category,
        unseen_type=unseen_type,
        normal_budget=normal_budget,
        anomaly_budget=anomaly_budget,
        public_val_fraction=0.2,
        seed=42,
    )
    print(f"[experiment] split={split_dir}")

    # ---- 2. pretrain-public（该类别专属 manifest，防泄漏）----
    if not skip_pretrain:
        train_records = load_jsonl(
            split_dir / "public_train.jsonl"
        )
        validation_records = load_jsonl(
            split_dir / "public_val.jsonl"
        )
        print(
            f"[experiment] pretrain: "
            f"train={len(train_records)}, "
            f"val={len(validation_records)}"
        )
        model = aoi_main.build_model(config)
        transfer = FewShotTransfer(config, model)
        transfer.pretrain_public(
            train_records=train_records,
            validation_records=validation_records,
            output_path=config.industrial_checkpoint_path,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        # 预训练后保留 model + transfer 直接进入 adapt 微调；
        # 旧代码在此 del 导致 adapt 处 UnboundLocalError。
        torch.cuda.empty_cache()
    else:
        if not config.industrial_checkpoint_path.exists():
            raise FileNotFoundError(
                f"--skip-pretrain 但预训练权重不存在：\n"
                f"{config.industrial_checkpoint_path}"
            )
        model = aoi_main.build_model(
            config,
            full_checkpoint=config.industrial_checkpoint_path,
        )
        transfer = FewShotTransfer(config, model)

    # ---- 3. adapt ----
    deployment_dir = experiment_root / "deployment"
    transfer.adapt(
        normal_paths=list_images(
            split_dir / "support" / "normal"
        ),
        anomaly_paths=list_images(
            split_dir / "support" / "anomaly"
        ),
        output_model_path=deployment_dir / "target_model.pth",
        output_reference_path=(
            deployment_dir / "normal_reference.pth"
        ),
        synthetic_engine=SyntheticEngine(seed=42),
    )
    del model, transfer
    torch.cuda.empty_cache()

    # ---- 4. evaluate（含 threshold sweep）----
    model2, reference = aoi_main.load_deployed(config)
    detector = AOIRealtimeDetector(
        config=config,
        model=model2,
        reference=reference,
    )
    metrics = detector.evaluate_dataset(
        normal_paths=list_images(split_dir / "query" / "normal"),
        seen_anomaly_paths=list_images(
            split_dir / "query" / "seen"
        ),
        unseen_anomaly_paths=list_images(
            split_dir / "query" / "unseen"
        ),
        output_dir=experiment_root / "evaluation",
        threshold_sweep=True,
    )
    del model2, reference, detector
    torch.cuda.empty_cache()

    # ---- 5. 可选可视化 ----
    if visualize:
        import vis_infer

        old_argv = sys.argv
        sys.argv = [
            "vis_infer.py",
            "--config",
            str(experiment_root / "config.json"),
            "--normal-dir",
            str(split_dir / "query" / "normal"),
            "--seen-dir",
            str(split_dir / "query" / "seen"),
            "--unseen-dir",
            str(split_dir / "query" / "unseen"),
            "--save-dir",
            str(experiment_root / "vis"),
            "--max-per-type",
            "6",
        ]
        try:
            vis_infer.main()
        finally:
            sys.argv = old_argv

    # ---- 6. 追加 summary.csv ----
    sweep = metrics.get("threshold_sweep", {})
    best_f1 = sweep.get("best_f1_point", {})
    p_at_fpr = next(
        (v for k, v in sweep.items() if k.startswith("precision_at_fpr")),
        {},
    )
    row = {
        "tag": tag,
        "target_dataset": target_dataset,
        "target_category": target_category,
        "unseen_type": unseen_type or "",
        "count": metrics["count"],
        "overall_auroc": metrics["overall_auroc"],
        "seen_auroc": metrics["seen_auroc"],
        "unseen_auroc": metrics["unseen_auroc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "normal_fpr": metrics["normal_fpr"],
        "seen_recall": metrics["seen_recall"],
        "unseen_recall": metrics["unseen_recall"],
        "deploy_threshold": metrics["deploy_threshold"],
        "best_f1_threshold": best_f1.get("threshold"),
        "best_f1_f1": best_f1.get("f1"),
        "p_at_target_fpr_threshold": p_at_fpr.get("threshold"),
        "p_at_target_fpr_precision": p_at_fpr.get("precision"),
        "p_at_target_fpr_recall": p_at_fpr.get("recall"),
        "mean_latency_ms": metrics["mean_latency_ms"],
        "p95_latency_ms": metrics["p95_latency_ms"],
        "wall_time_s": round(time.time() - started, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    summary_path = base.root_path / "experiments" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"[experiment] done tag={tag} "
        f"wall={row['wall_time_s']}s → {summary_path}"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified per-category AOI baseline runner"
    )
    parser.add_argument(
        "--base-config",
        default=str(PROJECT_ROOT / "config.json"),
        help="基础配置（workspace 等会被每类实验目录覆盖）",
    )
    parser.add_argument("--target-dataset", default=None)
    parser.add_argument("--target-category", default=None)
    parser.add_argument("--unseen-type", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None
    )
    parser.add_argument(
        "--normal-budget", type=int, default=100
    )
    parser.add_argument(
        "--anomaly-budget", type=int, default=30
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="复用该实验目录已有的 industrial_pretrained.pth",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="评估后跑 vis_infer 出对比图",
    )
    args = parser.parse_args()

    if args.all:
        for dataset, category, unseen in ALL_CATEGORIES:
            print(
                f"\n{'=' * 60}\n"
                f"[run-all] {dataset}/{category} "
                f"unseen={unseen}\n{'=' * 60}"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--base-config",
                args.base_config,
                "--target-dataset",
                dataset,
                "--target-category",
                category,
                "--epochs",
                str(args.epochs) if args.epochs else "0",
                "--steps-per-epoch",
                str(args.steps_per_epoch)
                if args.steps_per_epoch
                else "0",
                "--normal-budget",
                str(args.normal_budget),
                "--anomaly-budget",
                str(args.anomaly_budget),
            ]
            if unseen:
                command += [
                    "--unseen-type",
                    unseen,
                ]
            if args.skip_pretrain:
                command.append("--skip-pretrain")
            if args.visualize:
                command.append("--visualize")
            completed = subprocess.run(command, cwd=PROJECT_ROOT)
            if completed.returncode != 0:
                print(
                    f"[run-all] {category} 失败"
                    f"（exit={completed.returncode}），继续下一类。"
                )
        return

    if not (args.target_dataset and args.target_category):
        parser.error(
            "需要 --target-dataset + --target-category，"
            "或使用 --all"
        )

    run_experiment(
        base_config_path=args.base_config,
        target_dataset=args.target_dataset,
        target_category=args.target_category,
        unseen_type=args.unseen_type,
        epochs=args.epochs or None,
        steps_per_epoch=args.steps_per_epoch or None,
        normal_budget=args.normal_budget,
        anomaly_budget=args.anomaly_budget,
        skip_pretrain=args.skip_pretrain,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
