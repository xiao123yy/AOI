from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def normalize_image(
    image: Image.Image,
    size: int,
) -> torch.Tensor:
    image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def lab_statistics(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    means = lab.reshape(-1, 3).mean(axis=0)
    stds = lab.reshape(-1, 3).std(axis=0)

    histograms = []
    for channel in range(3):
        histogram = cv2.calcHist(
            [lab.astype(np.uint8)],
            [channel],
            None,
            [16],
            [0, 256],
        ).reshape(-1)
        histogram /= histogram.sum() + 1e-6
        histograms.append(histogram)

    return np.concatenate([means, stds, *histograms]).astype(
        np.float32
    )


def geometry_statistics(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB"), dtype=np.uint8),
        cv2.COLOR_RGB2GRAY,
    )
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros(6, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    return np.array(
        [
            width / max(1, image.width),
            height / max(1, image.height),
            x / max(1, image.width),
            y / max(1, image.height),
            area / max(1, image.width * image.height),
            perimeter / max(1, image.width + image.height),
        ],
        dtype=np.float32,
    )


def nms_boxes(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    iou_threshold: float = 0.35,
) -> list[int]:
    if not boxes:
        return []

    boxes_array = np.asarray(boxes, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    order = scores_array.argsort()[::-1]
    keep = []

    while order.size:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(
            boxes_array[current, 0], boxes_array[rest, 0]
        )
        yy1 = np.maximum(
            boxes_array[current, 1], boxes_array[rest, 1]
        )
        xx2 = np.minimum(
            boxes_array[current, 2], boxes_array[rest, 2]
        )
        yy2 = np.minimum(
            boxes_array[current, 3], boxes_array[rest, 3]
        )

        width = np.maximum(0, xx2 - xx1)
        height = np.maximum(0, yy2 - yy1)
        intersection = width * height

        area_current = (
            (boxes_array[current, 2] - boxes_array[current, 0])
            * (boxes_array[current, 3] - boxes_array[current, 1])
        )
        area_rest = (
            (boxes_array[rest, 2] - boxes_array[rest, 0])
            * (boxes_array[rest, 3] - boxes_array[rest, 1])
        )

        iou = intersection / (
            area_current + area_rest - intersection + 1e-6
        )
        order = rest[iou <= iou_threshold]

    return keep
