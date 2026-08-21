from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


DEFAULT_SCALE_FACTORS = (0.8, 0.9, 1.1, 1.2)
"""文档建议的尺度干预比例：各向同性与各向异性都会用。"""


def random_scale_pair(rng) -> tuple[float, float]:
    """抽一组 (sx, sy)：50% 各向同性、50% 各向异性，比例来自文档。"""
    factors = DEFAULT_SCALE_FACTORS
    try:
        sx = float(rng.choice(factors))
        sy = float(rng.choice(factors))
    except AttributeError:  # random.Random（无 np 风格 choice 情形）
        sx = float(rng.choice(factors))
        sy = float(rng.choice(factors))
    return sx, sy


def scale_intervene(
    image: Image.Image,
    sx: float,
    sy: float,
) -> Image.Image:
    """无黑边、保持中心、不改变画布尺寸的仿射缩放。

    等价于"镜头变焦/尺子实缩"：绕图像中心按 (sx, sy) 缩放，
    边缘内容回填（BORDER_REFLECT_101），避免模型学到"黑边=异常"捷径。
    （尺寸异常检测方法文档的"normal → scale 0.8/0.9/1.1/1.2，
     保持中心基本不变"的干预构造。）
    """
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    matrix = np.array(
        [
            [sx, 0.0, (1.0 - sx) * width / 2.0],
            [0.0, sy, (1.0 - sy) * height / 2.0],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(warped)


class SyntheticEngine:
    """
    迁移学习中的辅助数据引擎。
    GAN外观生成器可通过appearance_generator接口接入；
    当前模块同时提供尺寸、缺件/错位、颜色等可控合成。
    """

    def __init__(self, appearance_generator=None, seed: int = 42):
        self.appearance_generator = appearance_generator
        self.random = random.Random(seed)

    def appearance(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> tuple[Image.Image, Image.Image]:
        if self.appearance_generator is not None:
            return self.appearance_generator.generate(image, mask)

        # 无GAN时的保底外观异常：局部裂纹/污渍。
        array = np.asarray(image.convert("RGB")).copy()
        height, width = array.shape[:2]
        output_mask = np.zeros((height, width), dtype=np.uint8)

        start = (
            self.random.randint(0, width - 1),
            self.random.randint(0, height - 1),
        )
        end = (
            int(np.clip(start[0] + self.random.randint(-width // 4, width // 4), 0, width - 1)),
            int(np.clip(start[1] + self.random.randint(-height // 4, height // 4), 0, height - 1)),
        )
        thickness = self.random.randint(2, max(3, min(width, height) // 40))
        color = tuple(
            int(value)
            for value in np.random.randint(0, 90, size=3)
        )

        cv2.line(array, start, end, color, thickness)
        cv2.line(output_mask, start, end, 255, thickness * 2)

        return (
            Image.fromarray(array),
            Image.fromarray(output_mask),
        )

    def color(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Color(image).enhance(
            self.random.uniform(0.55, 1.55)
        )
        image = ImageEnhance.Brightness(image).enhance(
            self.random.uniform(0.75, 1.25)
        )
        return image

    def geometry(self, image: Image.Image) -> Image.Image:
        """[旧接口] 保留：整图缩放。新代码请用 scale_intervention（带标签）。"""
        array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]

        scale_x = self.random.uniform(0.85, 1.15)
        scale_y = self.random.uniform(0.85, 1.15)

        resized = cv2.resize(
            array,
            None,
            fx=scale_x,
            fy=scale_y,
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.zeros_like(array)
        new_height, new_width = resized.shape[:2]

        crop = resized[
            : min(new_height, height),
            : min(new_width, width),
        ]
        y0 = max(0, (height - crop.shape[0]) // 2)
        x0 = max(0, (width - crop.shape[1]) // 2)
        canvas[
            y0:y0 + crop.shape[0],
            x0:x0 + crop.shape[1],
        ] = crop

        return Image.fromarray(canvas)

    def scale_intervention(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, Image.Image, float, float]:
        """尺寸异常合成干预（文档方法）：返回 (干预图, 全图掩码, sx, sy)。

        对正常图做无黑边中心仿射缩放（iso/aniso、比例 0.8/0.9/1.1/1.2），
        同时给出 (sx, sy) 标签，供 scale_head 学习"相对尺度变化"。
        """
        sx, sy = random_scale_pair(self.random)
        warped = scale_intervene(image, sx, sy)
        height, width = warped.size
        mask = Image.fromarray(
            np.full((height, width), 255, dtype=np.uint8)
        )
        return warped, mask, sx, sy

    def component_missing(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> Image.Image:
        array = np.asarray(image.convert("RGB")).copy()
        x0, y0, x1, y1 = box

        mask = np.zeros(array.shape[:2], dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255

        inpainted = cv2.inpaint(
            cv2.cvtColor(array, cv2.COLOR_RGB2BGR),
            mask,
            5,
            cv2.INPAINT_TELEA,
        )
        return Image.fromarray(
            cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        )
