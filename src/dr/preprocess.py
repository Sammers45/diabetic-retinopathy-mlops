"""The single image module shared by data prep, training, serving and monitoring.

Everything that turns bytes into pixels or pixels into statistics lives here, so that the
training pipeline, the served pyfunc model and the drift monitor can never disagree (a previous
attempt had train/serve skew from decoding the same JPEG with two libraries).

Phase 2 contents: decoding, the fundus mask, per-image statistics and the quality flag.
Phase 3 adds cropping, padding, resizing and Ben Graham normalisation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Statistic columns produced by image_stats(); also the drift-monitoring image features.
STAT_COLUMNS: tuple[str, ...] = (
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
    "fundus_fraction",
)
QUALITY_OK = "ok"
POOR_REASONS = ("unreadable", "no_fundus", "too_dark", "overexposed", "blurred")


# --------------------------------------------------------------------------- decoding
def decode_image(data: bytes) -> np.ndarray | None:
    """Decode encoded image bytes to a BGR uint8 array, or ``None`` if undecodable.

    Decoding from bytes (not from a path) is the one code path used everywhere: files are read
    as bytes first, API uploads arrive as bytes, base64 inputs are decoded to bytes.
    """
    if not data:
        return None
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None or img.ndim != 3 or img.shape[0] < 8 or img.shape[1] < 8:
        return None
    return img


def read_bytes(path: str | Path) -> bytes:
    """Read a file as bytes. ``np.fromfile``-style reading also handles non-ASCII Windows paths."""
    return Path(path).read_bytes()


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image file into BGR uint8 (``None`` if missing or unreadable)."""
    try:
        return decode_image(read_bytes(path))
    except OSError:
        return None


def md5_bytes(data: bytes) -> str:
    """MD5 of the raw file bytes; used to group byte-identical files across patients."""
    return hashlib.md5(data).hexdigest()


# --------------------------------------------------------------------------- fundus mask
def resize_long_side(img: np.ndarray, size: int) -> np.ndarray:
    """Resize so the longer side equals ``size`` (area interpolation when shrinking)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale == 1.0:
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)


def fundus_mask(img: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    """Boolean mask of the circular fundus region.

    Why the max channel rather than grey: the fundus is dominated by red, and very dark
    photographs still separate from the black camera border in the red channel while their grey
    level is close to the border's. A median blur removes JPEG noise in the border first.
    """
    chan = img.max(axis=2)
    chan = cv2.medianBlur(chan, 5)
    mask = chan > threshold
    # Keep the largest connected component: removes text/labels burnt into the border.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest
    return mask


# --------------------------------------------------------------------------- statistics
def image_stats(img: np.ndarray | None, cfg_quality: Mapping[str, Any]) -> dict[str, Any]:
    """Per-image statistics computed at a fixed working resolution.

    All statistics except width/height are computed after resizing the long side to
    ``quality.sharpness_size`` (512) so they do not depend on the camera resolution; sharpness
    in particular (variance of the Laplacian) grows with resolution.
    """
    if img is None:
        return {c: np.nan for c in STAT_COLUMNS} | {"dark_p99": np.nan}
    h, w = img.shape[:2]
    small = resize_long_side(img, int(cfg_quality.get("sharpness_size", 512)))
    mask = fundus_mask(small, float(cfg_quality.get("mask_threshold", 10.0)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fraction = float(mask.mean())
    # 99th percentile of the max channel: separates "very dark" from "no fundus at all".
    dark_p99 = float(np.percentile(small.max(axis=2), 99))
    if mask.sum() < 50:
        return {
            "width": w,
            "height": h,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "sharpness": 0.0,
            "fundus_fraction": fraction,
            "dark_p99": dark_p99,
        }
    # Erode before measuring sharpness so the strong fundus/border edge does not count.
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharp_region = inner if inner.sum() >= 50 else mask
    return {
        "width": w,
        "height": h,
        "brightness": float(gray[mask].mean()),
        "contrast": float(gray[mask].std()),
        "sharpness": float(lap[sharp_region].var()),
        "fundus_fraction": fraction,
        "dark_p99": dark_p99,
    }


def quality_flag(stats: Mapping[str, Any], cfg_quality: Mapping[str, Any], readable: bool = True) -> str:
    """``ok`` or ``poor:<reason>``; the first failing check (in POOR_REASONS order) wins."""
    if not readable:
        return "poor:unreadable"
    if stats["fundus_fraction"] < float(cfg_quality["min_fundus_fraction"]):
        # A small mask on an under-exposed frame means "too dark" (the dark retina fell below the
        # mask threshold), not "no fundus": either the frame has no bright content at all, or the
        # part that was detected is itself dark.
        if stats["dark_p99"] < float(cfg_quality.get("dark_p99", 40.0)) or stats["brightness"] < float(
            cfg_quality["min_brightness"]
        ):
            return "poor:too_dark"
        return "poor:no_fundus"
    if stats["brightness"] < float(cfg_quality["min_brightness"]):
        return "poor:too_dark"
    if stats["brightness"] > float(cfg_quality["max_brightness"]):
        return "poor:overexposed"
    if stats["sharpness"] < float(cfg_quality["min_sharpness"]):
        return "poor:blurred"
    return QUALITY_OK


def analyse_bytes(data: bytes, cfg_quality: Mapping[str, Any]) -> dict[str, Any]:
    """Decode + statistics + quality flag + MD5 for one encoded image."""
    img = decode_image(data)
    stats = image_stats(img, cfg_quality)
    readable = img is not None
    return stats | {"md5": md5_bytes(data), "readable": readable, "quality": quality_flag(stats, cfg_quality, readable)}
