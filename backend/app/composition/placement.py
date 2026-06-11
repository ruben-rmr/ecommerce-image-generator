"""
Colocación automática: escalado y posicionamiento del objeto segmentado sobre el lienzo.
Por defecto es heurístico, pero se respeta el ajuste manual (coordenadas relativas).
"""

import cv2
import numpy as np

from .io_utils import compute_object_footprint, resize_rgba


def auto_scale(rgba: np.ndarray, canvas_size: tuple[int, int],
               target_height_ratio: float = 0.62) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Escala `rgba` para que la altura de la silueta sea `target_height_ratio` de la altura
    del lienzo. Se redimensiona la imagen completa de forma proporcional.

    Devuelve (rgba_redimensionado, (x_sup_izq, y_sup_izq)), donde la esquina superior
    izquierda por defecto centra el objeto en horizontal y apoya la silueta al 78 % de la
    altura del lienzo (línea de suelo típica de estudio).
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

    # Recalculamos la huella sobre la imagen ya redimensionada para que la colocación sea exacta.
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
    Detecta la línea de suelo/horizonte a partir del gradiente vertical de luminancia.
    Devuelve la y relativa en [0..1]. Si la señal es plana, recurre a 0.70.
    """
    gray = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape
    # Luminancia media por fila, suavizada.
    row_mean = gray.mean(axis=1)
    row_mean = cv2.GaussianBlur(row_mean.reshape(-1, 1), (0, 0), sigmaY=H * 0.02, sigmaX=0).ravel()
    grad = np.abs(np.diff(row_mean))
    if grad.size == 0 or grad.max() < 4.0:
        return 0.70
    # Sesgamos la búsqueda hacia la mitad inferior, que es donde suele estar el suelo.
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
    Escala y posiciona el objeto sobre el fondo de una escena.

    - target_height_ratio: altura de la silueta / altura del lienzo (lo sobrescribe manual_scale).
    - ground_y_rel: dónde caen los pies del objeto (0..1) por defecto.
    - manual_position: (rel_x, rel_y) del punto CENTRO-PIES del objeto en [0..1].
    - manual_scale: si se indica, sustituye a target_height_ratio.
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
    Devuelve un rectángulo del fondo tomado alrededor y debajo del objeto, que se usa como
    fuente para armonizar el color. Por defecto es una franja de 1.5x el ancho del objeto,
    centrada horizontalmente sobre él.
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
