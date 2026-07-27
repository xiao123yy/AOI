from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "path",
    "dataset",
    "category",
    "split",
    "label",
    "defect_type",
    "task_type",
}


def read_manifest(
    path: str | Path,
) -> pd.DataFrame:
    manifest_path = Path(path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest不存在：{manifest_path}"
        )

    frame = pd.read_csv(manifest_path)

    missing_columns = (
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{manifest_path}缺少字段："
            f"{sorted(missing_columns)}"
        )

    frame["path"] = (
        frame["path"]
        .fillna("")
        .astype(str)
    )

    if "mask_path" not in frame.columns:
        frame["mask_path"] = ""
    else:
        frame["mask_path"] = (
            frame["mask_path"]
            .fillna("")
            .astype(str)
        )

    frame["label"] = (
        frame["label"]
        .astype(int)
    )

    return frame


def paths_from_manifest(
    path: str | Path,
    label: int | None = None,
) -> list[str]:
    frame = read_manifest(path)

    if label is not None:
        frame = frame[
            frame["label"] == int(label)
        ]

    return frame["path"].tolist()