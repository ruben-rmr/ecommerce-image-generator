"""
Estimación de la dirección de la luz y reiluminación ligera (gradiente sobre el canal L en LAB).

Nunca repintamos la geometría del producto: solo añadimos un gradiente direccional sutil de
luminancia confinado a la silueta del objeto.
"""

import cv2
import numpy as np

from .io_utils import srgb_to_lab, lab_to_srgb


def estimate_light_dir(bg_rgb: np.ndarray) -> tuple[float, float]:
    """
    Estima la dirección de la luz localizando el centroide de la zona más brillante en una
    copia reducida y desenfocada del fondo.

    Devuelve un vector unitario (dx, dy). Si no encuentra señal, recurre a una luz principal
    suave arriba-derecha: (+0.4, -0.6). El eje y crece hacia abajo, así que una y negativa
    significa "desde arriba".
    """
    h, w = bg_rgb.shape[:2]
    scale = 128 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(bg_rgb, (max(int(w * scale), 8), max(int(h * scale), 8)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = bg_rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=8.0, sigmaY=8.0)

    p95 = float(np.percentile(gray, 95))
    mask = gray >= p95
    if not mask.any():
        return (0.4, -0.6)

    ys, xs = np.where(mask)
    cx = xs.mean()
    cy = ys.mean()
    H, W = gray.shape
    dx = (cx - W / 2.0) / (W / 2.0)
    dy = (cy - H / 2.0) / (H / 2.0)
    norm = float(np.hypot(dx, dy))
    if norm < 0.05:
        return (0.4, -0.6)
    return (float(dx / norm), float(dy / norm))


def _directional_gradient(shape: tuple[int, int], direction: tuple[float, float]) -> np.ndarray:
    """
    Construye un gradiente float32 HxW normalizado en [-1, 1] que crece a lo largo de `direction`.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dx, dy = direction
    norm = max(np.hypot(dx, dy), 1e-6)
    dx, dy = dx / norm, dy / norm
    proj = (xx - cx) * dx + (yy - cy) * dy
    proj /= max(np.hypot(w, h) / 2.0, 1.0)
    return np.clip(proj, -1.0, 1.0)


def apply_directional_relight(rgba: np.ndarray, light_dir: tuple[float, float],
                              amplitude: float = 5.0, mix: float = 0.20) -> np.ndarray:
    """
    Suma al objeto un gradiente ±L sutil alineado con `light_dir`. Solo afecta a la silueta
    (ponderado por el alfa). `amplitude` va en unidades del canal L de LAB y `mix` escala la
    mezcla final.
    """
    if mix <= 0 or amplitude <= 0:
        return rgba
    out = rgba.copy()
    alpha = out[..., 3]
    if not (alpha > 8).any():
        return out

    h, w = alpha.shape
    grad = _directional_gradient((h, w), light_dir)
    lab = srgb_to_lab(out[..., :3])
    weight = (alpha.astype(np.float32) / 255.0) * float(mix)
    lab[..., 0] = np.clip(lab[..., 0] + grad * amplitude * weight, 0, 255)
    out[..., :3] = lab_to_srgb(lab)
    return out


def apply_studio_keyfill(rgba: np.ndarray, amplitude: float = 6.0, mix: float = 0.25) -> np.ndarray:
    """
    Esquema key+fill de estudio: más claro arriba-izquierda y algo más oscuro abajo-derecha.
    La dirección es fija porque en las fotos de producto la luz suele venir desde arriba-izquierda.
    """
    return apply_directional_relight(rgba, (-0.6, -0.6), amplitude=amplitude, mix=mix)
