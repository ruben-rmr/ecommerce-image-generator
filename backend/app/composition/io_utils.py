"""I/O helpers and small color-space utilities used across the composition pipeline."""

import io
import numpy as np
import cv2
from PIL import Image


def png_bytes_to_rgba(data: bytes) -> np.ndarray:
    """Decode PNG bytes into an HxWx4 uint8 RGBA array."""
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.array(img)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an HxWx4 uint8 RGBA array into PNG bytes."""
    pil = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 RGB array into PNG bytes."""
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def load_background(path: str, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load a background JPG/PNG as RGB (HxWx3, uint8). EXIF orientation is honored.
    If target_size=(W,H) is given, the image is resized to that size.
    """
    pil = Image.open(path)
    try:
        from PIL import ImageOps
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if target_size is not None:
        pil = pil.resize(target_size, Image.LANCZOS)
    return np.array(pil)


def alpha_blend(fg_rgb: np.ndarray, fg_a: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """
    Standard premultiplied alpha-blend. fg_a in [0,255]. Returns uint8 RGB.
    All inputs must share the same HxW.
    """
    a = fg_a.astype(np.float32) / 255.0
    a3 = a[..., None]
    out = fg_rgb.astype(np.float32) * a3 + bg_rgb.astype(np.float32) * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


def multiply_shadow(bg_rgb: np.ndarray, shadow_alpha: np.ndarray, color=(0, 0, 0), opacity: float = 1.0) -> np.ndarray:
    """
    Darken bg_rgb by a soft shadow (HxW float[0..1] or uint8). The shadow color
    defaults to black; opacity scales the final effect. Returns uint8 RGB.
    """
    if shadow_alpha.dtype != np.float32:
        s = shadow_alpha.astype(np.float32) / 255.0
    else:
        s = shadow_alpha
    s = np.clip(s * float(opacity), 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = bg_rgb.astype(np.float32) * (1.0 - s) + color_arr * s
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_rgba_onto_canvas(canvas_rgb: np.ndarray, obj_rgba: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    """
    Composite obj_rgba (HxWx4) over canvas_rgb (HxWx3) at (x, y) top-left.
    Crops to canvas bounds. Returns uint8 RGB (modified copy).
    """
    H, W = canvas_rgb.shape[:2]
    oh, ow = obj_rgba.shape[:2]
    x, y = top_left

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return canvas_rgb.copy()

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = canvas_rgb[y1:y2, x1:x2].copy()
    obj_crop = obj_rgba[sy1:sy2, sx1:sx2]
    blended = alpha_blend(obj_crop[..., :3], obj_crop[..., 3], roi)

    out = canvas_rgb.copy()
    out[y1:y2, x1:x2] = blended
    return out


def compute_object_footprint(alpha: np.ndarray, threshold: int = 32) -> tuple[int, int, int, int]:
    """
    Returns (x_min, y_min, x_max, y_max) tight bbox of the alpha silhouette.
    Falls back to full image if no pixel exceeds threshold.
    """
    mask = alpha > threshold
    if not mask.any():
        h, w = alpha.shape[:2]
        return 0, 0, w - 1, h - 1
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_rgba(rgba: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx4 RGBA array preserving alpha (cv2 LANCZOS)."""
    return cv2.resize(rgba, new_size, interpolation=cv2.INTER_LANCZOS4)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 LAB (OpenCV scale: L 0-100, a/b ~-128..127)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """float32 LAB -> uint8 RGB."""
    lab8 = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB)
