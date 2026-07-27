from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import os
import random
import shutil

import cv2
import numpy as np

from config import AOIConfig


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def list_images(directory: str | Path) -> list[str]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)

    return sorted(
        str(path.resolve())
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _direct_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _recursive_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path.resolve()
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _find_mvtec_category_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []

    roots: dict[str, Path] = {}
    if (root / "train" / "good").exists():
        roots[str(root.resolve())] = root.resolve()

    for good_dir in root.rglob("good"):
        if (
            good_dir.is_dir()
            and good_dir.parent.name.lower() == "train"
        ):
            category_root = good_dir.parent.parent.resolve()
            roots[str(category_root)] = category_root

    return sorted(roots.values())


def _mvtec_mask_path(
    category_root: Path,
    defect_type: str,
    image_path: Path,
) -> str:
    candidates = [
        category_root
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}_mask.png",
        category_root
        / "ground_truth"
        / defect_type
        / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def scan_mvtec_like(
    root: str | Path,
    dataset_name: str,
    default_task: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for category_root in _find_mvtec_category_roots(Path(root)):
        category = category_root.name

        for image_path in _direct_images(
            category_root / "train" / "good"
        ):
            records.append({
                "path": str(image_path),
                "mask_path": "",
                "dataset": dataset_name,
                "category": category,
                "split": "train",
                "label": 0,
                "defect_type": "good",
                "task_type": default_task,
            })

        for split_name in ["validation", "test"]:
            split_root = category_root / split_name
            if not split_root.exists():
                continue

            for defect_dir in sorted(
                path for path in split_root.iterdir()
                if path.is_dir()
            ):
                defect_type = defect_dir.name
                normalized = defect_type.lower()
                label = int(
                    normalized not in {"good", "normal", "ok"}
                )

                if "logical" in normalized:
                    task_type = "logic"
                elif "structural" in normalized:
                    task_type = "structure"
                else:
                    task_type = default_task

                for image_path in _recursive_images(defect_dir):
                    records.append({
                        "path": str(image_path),
                        "mask_path": (
                            _mvtec_mask_path(
                                category_root,
                                defect_type,
                                image_path,
                            )
                            if label
                            else ""
                        ),
                        "dataset": dataset_name,
                        "category": category,
                        "split": split_name,
                        "label": label,
                        "defect_type": defect_type,
                        "task_type": task_type,
                    })

    return records


def _scan_visa_original(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    category_roots: set[Path] = set()

    for images_dir in root.rglob("Images"):
        if (
            images_dir.is_dir()
            and images_dir.parent.name == "Data"
        ):
            category_roots.add(images_dir.parent.parent.resolve())

    for category_root in sorted(category_roots):
        category = category_root.name
        image_root = category_root / "Data" / "Images"
        mask_root = category_root / "Data" / "Masks"

        for name in ["Normal", "normal"]:
            for image_path in _recursive_images(image_root / name):
                records.append({
                    "path": str(image_path),
                    "mask_path": "",
                    "dataset": "visa",
                    "category": category,
                    "split": "all",
                    "label": 0,
                    "defect_type": "good",
                    "task_type": "appearance",
                })

        for name in ["Anomaly", "anomaly"]:
            for image_path in _recursive_images(image_root / name):
                candidates = [
                    mask_root / "Anomaly" / image_path.name,
                    mask_root / "anomaly" / image_path.name,
                    mask_root / "Anomaly" / f"{image_path.stem}.png",
                    mask_root / "anomaly" / f"{image_path.stem}.png",
                ]
                mask_path = next(
                    (
                        str(candidate.resolve())
                        for candidate in candidates
                        if candidate.exists()
                    ),
                    "",
                )
                records.append({
                    "path": str(image_path),
                    "mask_path": mask_path,
                    "dataset": "visa",
                    "category": category,
                    "split": "all",
                    "label": 1,
                    "defect_type": "anomaly",
                    "task_type": "appearance",
                })

    return records


def scan_visa(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    converted = scan_mvtec_like(
        root_path,
        dataset_name="visa",
        default_task="appearance",
    )
    if converted:
        return converted
    return _scan_visa_original(root_path)


def _mask_nonempty(path: Path) -> bool:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return bool(mask is not None and np.any(mask > 0))


def scan_dagm(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    records: list[dict[str, Any]] = []

    class_dirs = sorted(
        {
            path.resolve()
            for path in root_path.rglob("Class*")
            if path.is_dir()
            and path.name.lower().startswith("class")
        }
    )

    for class_dir in class_dirs:
        category = class_dir.name

        for split_name in ["Train", "Test", "train", "test"]:
            split_dir = class_dir / split_name
            if not split_dir.exists():
                continue

            label_dirs = [
                split_dir / "Label",
                split_dir / "label",
                class_dir / f"{split_name}Label",
            ]
            label_dir = next(
                (path for path in label_dirs if path.exists()),
                None,
            )

            for image_path in _direct_images(split_dir):
                mask_path = ""
                if label_dir is not None:
                    candidates = [
                        label_dir
                        / f"{image_path.stem}_label{image_path.suffix}",
                        label_dir / f"{image_path.stem}_label.BMP",
                        label_dir / image_path.name,
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            mask_path = str(candidate.resolve())
                            break

                label = int(
                    bool(mask_path)
                    and _mask_nonempty(Path(mask_path))
                )
                records.append({
                    "path": str(image_path),
                    "mask_path": mask_path if label else "",
                    "dataset": "dagm2007",
                    "category": category,
                    "split": split_name.lower(),
                    "label": label,
                    "defect_type": "defect" if label else "good",
                    "task_type": "appearance",
                })

    return records


def scan_all_public_datasets(
    config: AOIConfig,
) -> list[dict[str, Any]]:
    groups = [
        scan_mvtec_like(
            config.mvtec_ad_root,
            dataset_name="mvtec_ad",
            default_task="appearance",
        ),
        scan_mvtec_like(
            config.mvtec_loco_root,
            dataset_name="mvtec_loco_ad",
            default_task="structure",
        ),
        scan_visa(config.visa_root),
        scan_dagm(config.dagm_root),
    ]

    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for group in groups:
        for record in group:
            path = str(Path(record["path"]).resolve())
            if path in seen_paths:
                continue
            seen_paths.add(path)
            record = dict(record)
            record["path"] = path
            record["category_key"] = (
                f"{record['dataset']}/{record['category']}"
            )
            records.append(record)

    if not records:
        raise RuntimeError(
            "No images were found in the four public datasets."
        )

    counts: dict[str, int] = defaultdict(int)
    categories: dict[str, set[str]] = defaultdict(set)
    for record in records:
        counts[record["dataset"]] += 1
        categories[record["dataset"]].add(record["category"])

    for dataset in sorted(counts):
        print(
            f"[scan] {dataset}: {counts[dataset]} images, "
            f"{len(categories[dataset])} categories"
        )

    return records


def save_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
    return output


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    records = []
    with source.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _balanced_take(
    records: list[dict[str, Any]],
    budget: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["defect_type"]].append(record)

    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < budget:
        moved = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                moved = True
                if len(selected) == budget:
                    break
        if not moved:
            break
    return selected


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_records(
    records: list[dict[str, Any]],
    destination: Path,
) -> None:
    _reset_directory(destination)

    for index, record in enumerate(records):
        source = Path(record["path"]).resolve()
        safe_defect = str(record["defect_type"]).replace("/", "_")
        name = (
            f"{index:05d}_{record['dataset']}_"
            f"{record['category']}_{safe_defect}_"
            f"{source.name}"
        )
        target = destination / name
        try:
            os.symlink(source, target)
        except OSError:
            shutil.copy2(source, target)


def build_public_and_target_split(
    config: AOIConfig,
    target_dataset: str,
    target_category: str,
    unseen_type: str,
    normal_budget: int = 100,
    anomaly_budget: int = 30,
    public_val_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    records = scan_all_public_datasets(config)
    rng = random.Random(seed)

    target = [
        record for record in records
        if record["dataset"] == target_dataset
        and record["category"] == target_category
    ]
    if not target:
        available = sorted(
            {
                record["category"]
                for record in records
                if record["dataset"] == target_dataset
            }
        )
        raise ValueError(
            f"Target {target_dataset}/{target_category} was not found. "
            f"Available categories: {available}"
        )

    public_pool = [
        record for record in records
        if not (
            record["dataset"] == target_dataset
            and record["category"] == target_category
        )
    ]

    category_keys_by_dataset: dict[str, list[str]] = defaultdict(list)
    for record in public_pool:
        key = record["category_key"]
        if key not in category_keys_by_dataset[record["dataset"]]:
            category_keys_by_dataset[record["dataset"]].append(key)

    validation_keys: set[str] = set()
    for dataset, keys in category_keys_by_dataset.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) > 1:
            count = max(1, round(len(keys) * public_val_fraction))
            count = min(count, len(keys) - 1)
            validation_keys.update(keys[:count])

    public_train = [
        record for record in public_pool
        if record["category_key"] not in validation_keys
    ]
    public_val = [
        record for record in public_pool
        if record["category_key"] in validation_keys
    ]

    train_normal = [
        record for record in target
        if record["label"] == 0
        and str(record["split"]).lower() == "train"
    ]
    query_normal = [
        record for record in target
        if record["label"] == 0
        and str(record["split"]).lower() in {"test", "validation"}
    ]
    if not train_normal:
        train_normal = [
            record for record in target if record["label"] == 0
        ]

    rng.shuffle(train_normal)
    support_normal = train_normal[:normal_budget]
    if len(support_normal) < normal_budget:
        print(
            f"[split] warning: requested {normal_budget} normal images, "
            f"only {len(support_normal)} are available."
        )

    anomalies = [record for record in target if record["label"] == 1]
    defect_types = sorted({record["defect_type"] for record in anomalies})
    if unseen_type not in defect_types:
        raise ValueError(
            f"Unseen defect type '{unseen_type}' was not found. "
            f"Available: {defect_types}"
        )

    query_unseen = [
        record for record in anomalies
        if record["defect_type"] == unseen_type
    ]
    seen_pool = [
        record for record in anomalies
        if record["defect_type"] != unseen_type
    ]
    support_anomaly = _balanced_take(
        seen_pool,
        anomaly_budget,
        rng,
    )
    support_ids = {record["path"] for record in support_anomaly}
    query_seen = [
        record for record in seen_pool
        if record["path"] not in support_ids
    ]

    experiment_name = (
        f"{target_dataset}_{target_category}_unseen_{unseen_type}"
    )
    split_dir = config.split_root / experiment_name
    split_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(public_train, split_dir / "public_train.jsonl")
    save_jsonl(public_val, split_dir / "public_val.jsonl")
    save_jsonl(support_normal, split_dir / "support_normal.jsonl")
    save_jsonl(support_anomaly, split_dir / "support_anomaly.jsonl")
    save_jsonl(query_normal, split_dir / "query_normal.jsonl")
    save_jsonl(query_seen, split_dir / "query_seen.jsonl")
    save_jsonl(query_unseen, split_dir / "query_unseen.jsonl")

    _link_records(support_normal, split_dir / "support" / "normal")
    _link_records(support_anomaly, split_dir / "support" / "anomaly")
    _link_records(query_normal, split_dir / "query" / "normal")
    _link_records(query_seen, split_dir / "query" / "seen")
    _link_records(query_unseen, split_dir / "query" / "unseen")

    summary = {
        "target_dataset": target_dataset,
        "target_category": target_category,
        "unseen_type": unseen_type,
        "public_train": len(public_train),
        "public_val": len(public_val),
        "support_normal": len(support_normal),
        "support_anomaly": len(support_anomaly),
        "query_normal": len(query_normal),
        "query_seen": len(query_seen),
        "query_unseen": len(query_unseen),
        "split_dir": str(split_dir),
    }
    (split_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return split_dir