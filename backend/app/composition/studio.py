"""
MODE 1 — Estudio profesional.

Procedural white/gray background + grounding shadows + light relighting.
No external assets, no IA. Target latency < 250 ms on 1024x1024.
"""

import time
import numpy as np
import cv2

from .io_utils import (
    png_bytes_to_rgba, rgb_to_png_bytes,
    alpha_blend, multiply_shadow,
    compute_object_footprint,
)
from .edges import prepare_object
from .placement import auto_scale
from .shadows import contact_shadow, cast_shadow
from .relighting import apply_studio_keyfill


VALID_STYLES = ("white", "soft_gray")


def _make_canvas(size: tuple[int, int], style: str) -> np.ndarray:
    W, H = size
    if style == "soft_gray":
        # Vertical gradient 250 -> 230.
        col = np.linspace(250, 230, H, dtype=np.float32).reshape(-1, 1)
        canvas = np.repeat(col, W, axis=1)
        canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    else:  # 'white' (default)
        canvas = np.full((H, W, 3), 250.0, dtype=np.float32)

    # Subtle radial vignette darkening the corners by ~4%.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / max(r.max(), 1.0)
    vignette = 1.0 - 0.04 * r_norm
    canvas = canvas * vignette[..., None]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def compose_studio(png_bytes: bytes,
                   style: str = "white",
                   canvas_size: tuple[int, int] = (1024, 1024),
                   manual_position: tuple[float, float] | None = None,
                   manual_scale: float | None = None,
                   target_height_ratio: float = 0.62) -> bytes:
    """
    Render the segmented object on a procedural studio background.

    Args:
        png_bytes: PNG RGBA of the segmented product.
        style:     'white' or 'soft_gray'.
        canvas_size: (W, H) of the output image.
        manual_position: optional (rel_cx, rel_y_feet) in [0..1]. Overrides default centering.
        manual_scale:    optional silhouette-height/canvas-height ratio. Overrides target_height_ratio.

    Returns: PNG bytes (RGB). The result is opaque (the studio background fills the frame).
    """
    if style not in VALID_STYLES:
        style = "white"

    t0 = time.perf_counter()
    rgba = png_bytes_to_rgba(png_bytes)
    rgba = prepare_object(rgba)

    if manual_scale is not None:
        target_height_ratio = float(np.clip(manual_scale, 0.05, 1.5))
    obj_resized, default_top_left = auto_scale(rgba, canvas_size, target_height_ratio)

    # Manual override of position uses the silhouette's center-feet anchor.
    if manual_position is not None:
        W, H = canvas_size
        fx_min, _, fx_max, fy_max = compute_object_footprint(obj_resized[..., 3])
        obj_cx = (fx_min + fx_max) // 2
        obj_y_bottom = fy_max
        cx_rel, cy_rel = manual_position
        target_cx = int(np.clip(cx_rel, 0.0, 1.0) * W)
        target_y_bottom = int(np.clip(cy_rel, 0.0, 1.0) * H)
        top_left = (target_cx - obj_cx, target_y_bottom - obj_y_bottom)
    else:
        top_left = default_top_left

    canvas = _make_canvas(canvas_size, style)

    # Nota: sin "ambient occlusion" alrededor de la silueta — generaba un halo de
    # sombreado rodeando todo el objeto. Solo conservamos sombras de suelo reales
    # (proyectada + contacto).
    drop = cast_shadow(canvas_size, obj_resized[..., 3], top_left,
                       light_dir=(-0.6, -0.6), length=0.45, squash=0.20,
                       fade=0.5, sigma_contact=4.0, sigma_tip=20.0, intensity=0.55)
    contact = contact_shadow(canvas_size, obj_resized[..., 3], top_left,
                             intensity=0.55, sigma=3.0, band_ratio=0.08)

    canvas = multiply_shadow(canvas, drop,    color=(0, 0, 0), opacity=1.0)

    obj_lit = apply_studio_keyfill(obj_resized, amplitude=6.0, mix=0.25)

    canvas = _paste_object(canvas, obj_lit, top_left)
    canvas = multiply_shadow(canvas, contact, color=(0, 0, 0), opacity=1.0)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"🎬 [studio:{style}] composed in {elapsed_ms:.0f} ms")
    return rgb_to_png_bytes(canvas)


def _paste_object(canvas_rgb: np.ndarray, obj_rgba: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    H, W = canvas_rgb.shape[:2]
    oh, ow = obj_rgba.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return canvas_rgb
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    crop = obj_rgba[sy1:sy2, sx1:sx2]
    blended = alpha_blend(crop[..., :3], crop[..., 3], canvas_rgb[y1:y2, x1:x2])
    out = canvas_rgb.copy()
    out[y1:y2, x1:x2] = blended
    return out
