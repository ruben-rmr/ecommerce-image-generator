"""
Auto-placement: scaling and positioning the segmented object on a canvas.
Heuristic by default; manual override (relative coordinates) is honored.
"""

import cv2
import numpy as np

from .io_utils import compute_object_footprint, resize_rgba


def auto_scale(rgba: np.ndarray, canvas_size: tuple[int, int],
               target_height_ratio: float = 0.62) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Scale `rgba` so the object's silhouette height equals `target_height_ratio`
    of the canvas height. The whole rgba image is resized proportionally.

    Returns (resized_rgba, (top_left_x_default, top_left_y_default)) where the
    default top-left centers the object horizontally and grounds the silhouette
    at 78 % of the canvas height (studio-style baseline).
    """
    W, H = canvas_size
    alpha = rgba[..., 3]
    x_min, y_min, x_max, y_max = compute_object_footprint(alpha)
    obj_h_px = max(y_max - y_min + 1, 1)
    obj_w_px = max(x_max - x_min + 1, 1)

    target_h_px = int(H * target_height_ratio)
    scale = target_h_px / obj_h_px
    new_w = max(int(rgba.shape[1] * scale), 1)
    new_h = max(int(rgba.shape[0] * scale), 1)

    resized = resize_rgba(rgba, (new_w, new_h))

    # Recompute footprint on the resized image so the placement is exact.
    rx_min, ry_min, rx_max, ry_max = compute_object_footprint(resized[..., 3])
    obj_cx = (rx_min + rx_max) // 2
    obj_y_bottom = ry_max

    target_cx = W // 2
    target_y_bottom = int(H * 0.78)
    top_left_x = target_cx - obj_cx
    top_left_y = target_y_bottom - obj_y_bottom

    return resized, (top_left_x, top_left_y)


def detect_ground_y(bg_rgb: np.ndarray) -> float:
    """
    Detect the ground/horizon line via vertical gradient of luminance.
    Returns relative y in [0..1]. Falls back to 0.70 when the signal is flat.
    """
    gray = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape
    # Row-wise mean luminance, smoothed.
    row_mean = gray.mean(axis=1)
    row_mean = cv2.GaussianBlur(row_mean.reshape(-1, 1), (0, 0), sigmaY=H * 0.02, sigmaX=0).ravel()
    grad = np.abs(np.diff(row_mean))
    if grad.size == 0 or grad.max() < 4.0:
        return 0.70
    # Bias the search toward the lower half (where ground usually is).
    weights = np.linspace(0.4, 1.2, grad.size, dtype=np.float32)
    score = grad * weights
    idx = int(np.argmax(score))
    return float(idx) / float(H)


def place_on_scene(rgba: np.ndarray, bg_size: tuple[int, int],
                   ground_y_rel: float,
                   target_height_ratio: float = 0.40,
                   manual_position: tuple[float, float] | None = None,
                   manual_scale: float | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Scale and position the object onto a scene background.

    - target_height_ratio: silhouette height / canvas height (overridden by manual_scale).
    - ground_y_rel: where the object's feet should land (0..1) by default.
    - manual_position: (rel_x, rel_y) for the object's CENTER-FEET point in [0..1].
    - manual_scale:    overrides target_height_ratio if provided.
    """
    W, H = bg_size
    if manual_scale is not None:
        target_height_ratio = float(np.clip(manual_scale, 0.05, 1.5))

    alpha = rgba[..., 3]
    x_min, y_min, x_max, y_max = compute_object_footprint(alpha)
    obj_h_px = max(y_max - y_min + 1, 1)

    target_h_px = max(int(H * target_height_ratio), 1)
    scale = target_h_px / obj_h_px
    new_w = max(int(rgba.shape[1] * scale), 1)
    new_h = max(int(rgba.shape[0] * scale), 1)
    resized = resize_rgba(rgba, (new_w, new_h))

    rx_min, ry_min, rx_max, ry_max = compute_object_footprint(resized[..., 3])
    obj_cx = (rx_min + rx_max) // 2
    obj_y_bottom = ry_max

    if manual_position is not None:
        cx_rel, cy_rel = manual_position
        target_cx = int(np.clip(cx_rel, 0.0, 1.0) * W)
        target_y_bottom = int(np.clip(cy_rel, 0.0, 1.0) * H)
    else:
        target_cx = W // 2
        target_y_bottom = int(np.clip(ground_y_rel, 0.05, 0.98) * H)

    top_left = (target_cx - obj_cx, target_y_bottom - obj_y_bottom)
    return resized, top_left


def background_roi_below_object(bg_rgb: np.ndarray, top_left: tuple[int, int],
                                 obj_size: tuple[int, int],
                                 footprint: tuple[int, int, int, int],
                                 expand: float = 0.5) -> np.ndarray:
    """
    Return a rectangle of the background sampled around/under the object,
    used as the source for color harmonization. Defaults to a strip 1.5x the
    object's width, centered horizontally on the object.
    """
    H, W = bg_rgb.shape[:2]
    ow, oh = obj_size
    x, y = top_left
    fx_min, fy_min, fx_max, fy_max = footprint

    obj_cx = x + (fx_min + fx_max) // 2
    obj_w  = max(fx_max - fx_min + 1, 1)
    obj_y_bottom = y + fy_max

    half_w = int(obj_w * (1.0 + expand) / 2.0)
    x1 = max(0, obj_cx - half_w)
    x2 = min(W, obj_cx + half_w)
    y1 = max(0, obj_y_bottom - int(0.2 * oh))
    y2 = min(H, obj_y_bottom + int(0.4 * oh))

    if x2 <= x1 or y2 <= y1:
        return bg_rgb.copy()
    return bg_rgb[y1:y2, x1:x2].copy()
