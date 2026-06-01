"""
MODE 2 — Escena comercial.

Composes the segmented PNG over a local background image with light/color
harmonization, atmospheric edge blending, directional shadow, optional
reflection. No IA generativa — only OpenCV/NumPy/Pillow.
"""

import time
import numpy as np

from .io_utils import (
    png_bytes_to_rgba, rgb_to_png_bytes,
    load_background, alpha_blend, multiply_shadow,
    compute_object_footprint,
)
from .edges import prepare_object
from .placement import place_on_scene, detect_ground_y, background_roi_below_object
from .harmonization import harmonize_lab, brightness_contrast_match, atmospheric_blend
from .relighting import estimate_light_dir, apply_directional_relight
from .shadows import contact_shadow, cast_shadow
from .reflections import make_reflection, reflection_top_left


def compose_scene(png_bytes: bytes,
                  bg_path: str,
                  metadata: dict | None = None,
                  canvas_size: tuple[int, int] | None = None,
                  manual_position: tuple[float, float] | None = None,
                  manual_scale: float | None = None,
                  harmonize_strength: float = 0.45,
                  target_height_ratio: float = 0.40) -> bytes:
    """
    Compose the segmented object onto a chosen background.

    Args:
        png_bytes:           PNG RGBA of the segmented product.
        bg_path:             absolute path to the background JPG/PNG.
        metadata:            optional dict with 'ground_y', 'light_dir',
                             'reflective' (bool), 'reflective_type' ('matte'|'glossy').
        canvas_size:         optional (W, H) of the output. Defaults to background size.
        manual_position:     optional (rel_cx, rel_y_feet) override.
        manual_scale:        optional silhouette-height ratio override.
        harmonize_strength:  Reinhard transfer strength on a/b channels (0..1).

    Returns: PNG bytes (RGB).
    """
    t0 = time.perf_counter()
    metadata = metadata or {}

    bg = load_background(bg_path, target_size=canvas_size)
    H, W = bg.shape[:2]
    canvas_size = (W, H)

    # 1) Object cleanup (halo decontamination + micro-feather).
    rgba = png_bytes_to_rgba(png_bytes)
    rgba = prepare_object(rgba)

    # 2) Auto-placement / manual override.
    ground_y = float(metadata.get("ground_y", detect_ground_y(bg)))
    obj_placed, top_left = place_on_scene(
        rgba, canvas_size,
        ground_y_rel=ground_y,
        target_height_ratio=target_height_ratio,
        manual_position=manual_position,
        manual_scale=manual_scale,
    )

    # 3) Light direction (metadata > estimate).
    if "light_dir" in metadata and isinstance(metadata["light_dir"], (list, tuple)) and len(metadata["light_dir"]) == 2:
        light_dir = (float(metadata["light_dir"][0]), float(metadata["light_dir"][1]))
    else:
        light_dir = estimate_light_dir(bg)

    # 4) Color harmonization (LAB Reinhard, partial), against a ROI under the object.
    footprint = compute_object_footprint(obj_placed[..., 3])
    bg_roi = background_roi_below_object(bg, top_left, obj_placed.shape[1::-1], footprint, expand=0.5)
    obj_harm = harmonize_lab(obj_placed, bg_roi, strength=float(harmonize_strength))
    obj_harm = brightness_contrast_match(obj_harm, bg_roi, strength=0.30)

    # 5) Directional relight (subtle).
    obj_harm = apply_directional_relight(obj_harm, light_dir, amplitude=5.0, mix=0.20)

    # 6) Atmospheric blending on the seam ring (against the background BEHIND the object).
    obj_harm = atmospheric_blend(bg, top_left, obj_harm,
                                 ring_thickness=3, bg_blur_sigma=3.0, mix=0.45)

    # 7) Compose canvas: shadows under, then object, then reflection above shadows.
    canvas = bg.copy()

    drop = cast_shadow(canvas_size, obj_harm[..., 3], top_left,
                       light_dir=light_dir, length=0.55, squash=0.28,
                       fade=0.5, sigma_contact=5.0, sigma_tip=26.0, intensity=0.55)
    canvas = multiply_shadow(canvas, drop, color=(0, 0, 0), opacity=1.0)

    # Optional reflection (matte/glossy).
    reflective = bool(metadata.get("reflective", False))
    if reflective:
        ref_type = str(metadata.get("reflective_type", "matte")).lower()
        blur_sigma = 1.0 if ref_type == "glossy" else 4.0
        fade = 0.30 if ref_type == "matte" else 0.45
        reflection = make_reflection(obj_harm, fade=fade, blur_sigma=blur_sigma)
        _, _, _, fy_max = footprint
        rx, ry = reflection_top_left(top_left, obj_harm.shape[1::-1], feet_y_local=fy_max)
        canvas = _paste_rgba(canvas, reflection, (rx, ry))

    # Object itself.
    canvas = _paste_rgba(canvas, obj_harm, top_left)

    # Sharp contact shadow on top to ground the object.
    contact = contact_shadow(canvas_size, obj_harm[..., 3], top_left,
                             intensity=0.55, sigma=3.0, band_ratio=0.08)
    canvas = multiply_shadow(canvas, contact, color=(0, 0, 0), opacity=1.0)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"🌅 [scene] {bg_path} ({W}x{H}) composed in {elapsed_ms:.0f} ms  light={light_dir}")
    return rgb_to_png_bytes(canvas)


def _paste_rgba(canvas_rgb: np.ndarray, obj_rgba: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
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
