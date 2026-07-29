"""
AOI Inference Visualization

Usage:
    # Single image
    python vis_infer.py --image data/mvtec_ad/grid/test/bent/000.png

    # Multiple images
    python vis_infer.py --image data/mvtec_ad/grid/test/bent/000.png data/mvtec_ad/grid/test/good/000.png --save-dir vis_results

    # Batch by category
    python vis_infer.py \
        --normal-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/normal \
        --seen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/seen \
        --unseen-dir aoi_full_workspace/splits/mvtec_ad_grid_unseen_bent/query/unseen \
        --save-dir vis_results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

# ── Project modules ─────────────────────────────────────────────
from config import AOIConfig
from aoi_model import AOIMultiBranchModel
from modules.normal_reference import NormalReference
from modules.realtime_detection import AOIRealtimeDetector
from utils.image import load_rgb, normalize_image


# ── Colors ────────────────────────────────────────────────────
COLOR_NORMAL = (76, 175, 80)
COLOR_ANOMALY = (244, 67, 54)
COLOR_BG = (33, 33, 33)
COLOR_TEXT = (255, 255, 255)


def load_deployed(config: AOIConfig) -> Tuple[AOIMultiBranchModel, NormalReference]:
    """Load deployed model and normal reference"""
    model_path = config.workspace_path / "deployment" / "target_model.pth"
    reference_path = config.workspace_path / "deployment" / "normal_reference.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference not found: {reference_path}")

    model = AOIMultiBranchModel(
        student_checkpoint=config.student_checkpoint,
        component_slots=config.component_slots,
        geometry_dims=config.geometry_dims,
        local_top_ratio=config.local_top_ratio,
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(config.device).eval()

    reference = NormalReference.load(reference_path, config)
    return model, reference


def get_heatmap(model, image: Image.Image, config: AOIConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract anomaly heatmap and feature activation maps"""
    tensor = normalize_image(image, config.global_size)[None].to(config.device)
    with torch.inference_mode():
        output = model(tensor)

    # Local anomaly heatmap (sigmoid 0~1)
    heatmap = torch.sigmoid(output["local_logits"][0, 0]).float().cpu().numpy()

    # F16 feature activation (max over channels)
    f16 = output["features"]["f16"][0].float().cpu().numpy()
    f16_activation = np.abs(f16).max(axis=0)

    return heatmap, f16_activation, output


def draw_info_panel(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: List[Tuple[str, str]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw info panel with label-value pairs"""
    line_height = font.size + 6
    panel_width = 380
    padding = 8

    # Background
    draw.rectangle(
        [x, y, x + panel_width, y + len(lines) * line_height + padding * 2],
        fill=(*COLOR_BG, 200),
    )

    for i, (label, value) in enumerate(lines):
        text_y = y + padding + i * line_height
        draw.text((x + padding, text_y), label, fill=COLOR_TEXT, font=font)
        draw.text((x + padding + 130, text_y), value, fill=COLOR_TEXT, font=font)


def visualize_single(
    image_path: str,
    detector: AOIRealtimeDetector,
    config: AOIConfig,
    save_dir: Optional[str] = None,
) -> Optional[str]:
    """Visualize inference result for a single image"""
    image = load_rgb(image_path)
    orig_w, orig_h = image.size

    # Run detector
    result = detector.inspect_image(image_path)

    # Get heatmap
    heatmap, f16_activation, model_output = get_heatmap(detector.model, image, config)

    # Resize to original image size
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    f16_resized = cv2.resize(f16_activation, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas: top row = 3 panels, bottom = info + bar chart
    info_height = 180
    canvas_width = orig_w * 3 + 20
    canvas_height = orig_h + info_height + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Original image
    canvas.paste(image, (0, 0))

    # Heatmap overlay
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = Image.blend(
        image, Image.fromarray(heatmap_colored), 0.45
    )
    canvas.paste(overlay, (orig_w + 10, 0))

    # F16 feature activation
    f16_norm = (f16_resized - f16_resized.min()) / (f16_resized.max() - f16_resized.min() + 1e-8)
    f16_colored = cv2.applyColorMap(
        (f16_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    f16_colored = cv2.cvtColor(f16_colored, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(f16_colored), (orig_w * 2 + 20, 0))

    # Column titles
    labels = ["Original Image", "Anomaly Heatmap", "F16 Activation"]
    for i, label in enumerate(labels):
        lx = orig_w * i + (10 if i > 0 else 0)
        draw.text((lx + 8, orig_h + 4), label, fill=(200, 200, 200), font=font_small)

    # Info panel
    is_anomaly = result["is_anomaly"]
    status_text = "ANOMALY" if is_anomaly else "NORMAL"

    branch = result.get("branch_scores", {})
    dominant = result.get("dominant_branch", "—")

    info_lines = [
        ("Image", Path(result["image"]).name),
        ("Status", status_text),
        ("Score", f"{result['score']:.4f}"),
        ("Threshold", f"{result['threshold']:.4f}"),
        ("Dominant", dominant),
        ("Latency", f"{result['latency_ms']:.1f} ms"),
    ]

    info_y = orig_h + 24
    draw_info_panel(draw, 8, info_y, info_lines, font_small)

    # ── Branch score bar chart ────────────────────────────────
    bar_x_start = 280
    bar_y = info_y
    bar_w = 28
    max_bar_h = 100
    bar_gap = 8

    bar_items = [
        ("Supervised", branch.get("supervised_global", 0)),
        ("LocalMem", branch.get("memory_local", 0)),
        ("GlobalMem", branch.get("memory_global", 0)),
        ("Color", branch.get("color", 0)),
        ("Geometry", branch.get("geometry", 0)),
    ]

    # Find max absolute value for scaling
    max_val = max(abs(v) for _, v in bar_items) if bar_items else 1.0
    max_val = max(max_val, 0.1)

    for bi, (bname, bval) in enumerate(bar_items):
        bx = bar_x_start + bi * (bar_w + bar_gap)
        bheight = int(abs(bval) / max_val * max_bar_h)
        bheight = max(2, min(bheight, max_bar_h))
        bcolor = (100, 200, 255) if bval >= 0 else (255, 180, 100)
        by0 = bar_y + max_bar_h - bheight
        draw.rectangle([bx, by0, bx + bar_w, bar_y + max_bar_h], fill=bcolor)
        # Score value
        draw.text(
            (bx, bar_y + max_bar_h + 4),
            f"{bval:.1f}",
            fill=(200, 200, 200),
            font=font_small,
        )
        # Label
        draw.text((bx, bar_y + max_bar_h + 22), bname, fill=(180, 180, 180), font=font_small)

    # ── Save ──────────────────────────────────────────────────
    save_path: Optional[str] = None
    stem = Path(image_path).stem
    parent_dir = Path(image_path).parent.name
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{parent_dir}_{stem}_result.png")
        canvas.save(save_path)
        print(f"[vis] Saved: {save_path}")
    else:
        save_path = os.path.join(os.path.dirname(image_path), f"{stem}_result.png")
        canvas.save(save_path)
        print(f"[vis] Saved: {save_path}")

    return save_path


def visualize_batch(
    image_paths: List[str],
    detector: AOIRealtimeDetector,
    config: AOIConfig,
    save_dir: str = "vis_results",
) -> None:
    """Batch visualization"""
    print(f"[vis] Processing {len(image_paths)} images, saving to {save_dir}/")
    for path in image_paths:
        try:
            visualize_single(path, detector, config, save_dir)
        except Exception as e:
            print(f"[vis] Failed: {path} — {e}", file=sys.stderr)


def list_images(directory: str) -> List[str]:
    """List all images in a directory recursively"""
    SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(
        str(p.resolve())
        for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def main():
    parser = argparse.ArgumentParser(description="AOI Inference Visualization")

    # Single / multiple images
    parser.add_argument("--image", nargs="+", default=None, help="Image path(s)")

    # Batch by category
    parser.add_argument("--normal-dir", default=None, help="Normal images directory")
    parser.add_argument("--seen-dir", default=None, help="Seen anomaly images directory")
    parser.add_argument("--unseen-dir", default=None, help="Unseen anomaly images directory")

    parser.add_argument("--save-dir", default="vis_results", help="Output directory")
    parser.add_argument("--max-per-type", type=int, default=6, help="Max images per category")
    parser.add_argument("--config", default="config.json", help="Config file path")

    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────
    config = AOIConfig.load(args.config)
    print("[vis] Loading deployment model...")
    model, reference = load_deployed(config)
    detector = AOIRealtimeDetector(config=config, model=model, reference=reference)

    # ── Collect image paths ─────────────────────────────────────
    image_paths: List[str] = []

    if args.image:
        image_paths.extend(args.image)

    for dir_path, label in [
        (args.normal_dir, "normal"),
        (args.seen_dir, "seen"),
        (args.unseen_dir, "unseen"),
    ]:
        if dir_path:
            paths = list_images(dir_path)
            selected = paths[: args.max_per_type]
            print(f"[vis] {label}: {len(paths)} images, showing top {len(selected)}")
            image_paths.extend(selected)

    if not image_paths:
        print("[vis] Please specify --image or --normal-dir/--seen-dir/--unseen-dir")
        sys.exit(1)

    # ── Run ──────────────────────────────────────────────────
    for path in image_paths:
        try:
            visualize_single(path, detector, config, args.save_dir)
        except Exception as e:
            print(f"[vis] Failed: {path} — {e}", file=sys.stderr)

    print(f"[vis] Done! Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
