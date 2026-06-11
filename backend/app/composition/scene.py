"""
MODO 2 — Escena comercial.

Compone el PNG segmentado sobre una imagen de fondo local, armonizando luz y color, fundiendo
el borde con la atmósfera del fondo, añadiendo una sombra direccional y, opcionalmente, un
reflejo. Sin IA generativa: solo OpenCV/NumPy/Pillow.
"""

import time
import numpy as np

from .io_utils import (
    png_bytes_to_rgba, rgb_to_png_bytes,
    load_background, alpha_blend, multiply_shadow,
    compute_object_footprint, print_stage_timings,
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
    Compone el objeto segmentado sobre el fondo elegido.

    png_bytes es el PNG RGBA del producto segmentado y bg_path la ruta absoluta del fondo
    JPG/PNG. `metadata` es un dict opcional con 'ground_y', 'light_dir', 'reflective' (bool) y
    'reflective_type' ('matte'|'glossy'). canvas_size fija el (W, H) de salida y, si se omite,
    se usa el tamaño del fondo. manual_position (rel_cx, rel_y_pies) y manual_scale permiten
    anular la colocación automática. harmonize_strength es la fuerza de la transferencia sobre
    los canales a/b (0..1).

    Devuelve los bytes de un PNG (RGB).
    """
    t0 = time.perf_counter()
    timings: list[tuple[str, float]] = []
    metadata = metadata or {}

    # 1) Carga del fondo.
    t = time.perf_counter()
    bg = load_background(bg_path, target_size=canvas_size)
    H, W = bg.shape[:2]
    canvas_size = (W, H)
    timings.append(("Carga del fondo", (time.perf_counter() - t) * 1000.0))

    # 2) Limpieza del objeto (descontaminación de halo + micro-suavizado).
    t = time.perf_counter()
    rgba = png_bytes_to_rgba(png_bytes)
    rgba = prepare_object(rgba)
    timings.append(("Limpieza del objeto", (time.perf_counter() - t) * 1000.0))

    # 3) Posicionamiento (colocación automática o manual + dirección de la luz).
    t = time.perf_counter()
    ground_y = float(metadata.get("ground_y", detect_ground_y(bg)))
    obj_placed, top_left = place_on_scene(
        rgba, canvas_size,
        ground_y_rel=ground_y,
        target_height_ratio=target_height_ratio,
        manual_position=manual_position,
        manual_scale=manual_scale,
    )
    # Dirección de la luz: la de los metadatos tiene prioridad sobre la estimada.
    if "light_dir" in metadata and isinstance(metadata["light_dir"], (list, tuple)) and len(metadata["light_dir"]) == 2:
        light_dir = (float(metadata["light_dir"][0]), float(metadata["light_dir"][1]))
    else:
        light_dir = estimate_light_dir(bg)
    timings.append(("Posicionamiento", (time.perf_counter() - t) * 1000.0))

    # 4) Armonización LAB (Reinhard parcial) contra un ROI bajo el objeto.
    t = time.perf_counter()
    footprint = compute_object_footprint(obj_placed[..., 3])
    bg_roi = background_roi_below_object(bg, top_left, obj_placed.shape[1::-1], footprint, expand=0.5)
    obj_harm = harmonize_lab(obj_placed, bg_roi, strength=float(harmonize_strength))
    obj_harm = brightness_contrast_match(obj_harm, bg_roi, strength=0.30)
    timings.append(("Armonización LAB", (time.perf_counter() - t) * 1000.0))

    # 5) Reiluminación direccional (sutil).
    t = time.perf_counter()
    obj_harm = apply_directional_relight(obj_harm, light_dir, amplitude=5.0, mix=0.20)
    timings.append(("Reiluminación direccional", (time.perf_counter() - t) * 1000.0))

    # 6) Fusión atmosférica en el anillo de la costura (contra el fondo que queda DETRÁS del objeto).
    t = time.perf_counter()
    obj_harm = atmospheric_blend(bg, top_left, obj_harm,
                                 ring_thickness=3, bg_blur_sigma=3.0, mix=0.45)
    timings.append(("Fusión atmosférica", (time.perf_counter() - t) * 1000.0))

    # Montaje del lienzo: primero las sombras, luego el objeto y, encima de las sombras, el reflejo.
    canvas = bg.copy()

    # 7) Sombra proyectada.
    t = time.perf_counter()
    drop = cast_shadow(canvas_size, obj_harm[..., 3], top_left,
                       light_dir=light_dir, length=0.55, squash=0.28,
                       fade=0.5, sigma_contact=5.0, sigma_tip=26.0, intensity=0.55)
    canvas = multiply_shadow(canvas, drop, color=(0, 0, 0), opacity=1.0)
    timings.append(("Sombra proyectada", (time.perf_counter() - t) * 1000.0))

    # 8) Pegado del objeto (primero el reflejo opcional y después el objeto en sí).
    t = time.perf_counter()
    reflective = bool(metadata.get("reflective", False))
    if reflective:
        ref_type = str(metadata.get("reflective_type", "matte")).lower()
        blur_sigma = 1.0 if ref_type == "glossy" else 4.0
        fade = 0.30 if ref_type == "matte" else 0.45
        reflection = make_reflection(obj_harm, fade=fade, blur_sigma=blur_sigma)
        _, _, _, fy_max = footprint
        rx, ry = reflection_top_left(top_left, obj_harm.shape[1::-1], feet_y_local=fy_max)
        canvas = _paste_rgba(canvas, reflection, (rx, ry))
    canvas = _paste_rgba(canvas, obj_harm, top_left)
    timings.append(("Pegado del objeto", (time.perf_counter() - t) * 1000.0))

    # 9) Sombra de contacto (nítida, por encima, para asentar el objeto en el suelo).
    t = time.perf_counter()
    contact = contact_shadow(canvas_size, obj_harm[..., 3], top_left,
                             intensity=0.55, sigma=3.0, band_ratio=0.08)
    canvas = multiply_shadow(canvas, contact, color=(0, 0, 0), opacity=1.0)
    timings.append(("Sombra de contacto", (time.perf_counter() - t) * 1000.0))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print_stage_timings(
        f"[scene] {bg_path} ({W}x{H})  light={light_dir}",
        timings, elapsed_ms, total_label="Total escena",
    )
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
