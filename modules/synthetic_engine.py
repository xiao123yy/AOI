from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


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
