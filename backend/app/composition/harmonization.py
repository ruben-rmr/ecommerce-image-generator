"""
Armonización de color/luminancia: transferencia LAB tipo Reinhard, ajuste de brillo/contraste
y fusión atmosférica del borde. Está acotada a propósito para preservar la identidad del producto.
"""

import cv2
import numpy as np

from .io_utils import srgb_to_lab, lab_to_srgb
from .edges import edge_ring_mask


def _masked_mean_std(channel: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Media y desviación de un canal restringido a mask>0. Si la máscara está vacía, usa toda la imagen."""
    sel = mask > 0
    if not sel.any():
        return float(channel.mean()), float(channel.std() + 1e-6)
    vals = channel[sel]
    return float(vals.mean()), float(vals.std() + 1e-6)


def harmonize_lab(obj_rgba: np.ndarray, bg_roi_rgb: np.ndarray,
                  strength: float = 0.45,
                  L_strength: float | None = None) -> np.ndarray:
    """
    Transferencia de color en LAB al estilo Reinhard, aplicada parcialmente.

    Los canales a/b usan `strength` (0.45 por defecto). El canal L usa `L_strength` (por
    defecto la mitad de `strength`) para no apagar el volumen ni las luces del producto.

    Solo se transforma el RGB del interior de la silueta (alfa > 8); el canal alfa no se toca.
    """
    if L_strength is None:
        L_strength = strength * 0.5

    rgba = obj_rgba.copy()
    alpha = rgba[..., 3]
    obj_mask = (alpha > 8).astype(np.uint8) * 255

    obj_lab = srgb_to_lab(rgba[..., :3])
    bg_lab  = srgb_to_lab(bg_roi_rgb)

    out_lab = obj_lab.copy()
    bg_mask = np.full(bg_lab.shape[:2], 255, dtype=np.uint8)

    for ch, s in zip(range(3), (L_strength, strength, strength)):
        if s <= 0:
            continue
        mu_o, sd_o = _masked_mean_std(obj_lab[..., ch], obj_mask)
        mu_b, sd_b = _masked_mean_std(bg_lab[..., ch],  bg_mask)
        ratio = float(np.clip(sd_b / max(sd_o, 1e-6), 0.5, 1.8))
        target = (obj_lab[..., ch] - mu_o) * ratio + mu_b
        out_lab[..., ch] = obj_lab[..., ch] * (1.0 - s) + target * s

    rgba[..., :3] = lab_to_srgb(out_lab)
    return rgba


def brightness_contrast_match(obj_rgba: np.ndarray, bg_roi_rgb: np.ndarray,
                              strength: float = 0.30) -> np.ndarray:
    """
    Ajuste fino que iguala solo la media del canal L (luminancia). Va bien después de
    harmonize_lab para cerrar cualquier diferencia de brillo que quede. La intensidad se
    mantiene en valores pequeños.
    """
    rgba = obj_rgba.copy()
    alpha = rgba[..., 3]
    obj_mask = (alpha > 8).astype(np.uint8) * 255

    obj_lab = srgb_to_lab(rgba[..., :3])
    bg_lab  = srgb_to_lab(bg_roi_rgb)
    mu_o, _ = _masked_mean_std(obj_lab[..., 0], obj_mask)
    mu_b    = float(bg_lab[..., 0].mean())
    delta   = (mu_b - mu_o) * float(strength)
    obj_lab[..., 0] = np.clip(obj_lab[..., 0] + delta, 0, 255)

    rgba[..., :3] = lab_to_srgb(obj_lab)
    return rgba


def atmospheric_blend(canvas_rgb: np.ndarray, top_left: tuple[int, int],
                      obj_rgba: np.ndarray,
                      ring_thickness: int = 3,
                      bg_blur_sigma: float = 3.0,
                      mix: float = 0.45) -> np.ndarray:
    """
    Mezcla el anillo del borde del objeto con una copia desenfocada del fondo justo debajo,
    ANTES de la mezcla alfa. Esto elimina el aspecto de "pegatina" en la costura.

    Devuelve un obj_rgba modificado (el canal alfa se deja intacto).
    """
    H, W = canvas_rgb.shape[:2]
    out = obj_rgba.copy()
    alpha = out[..., 3]
    ring = edge_ring_mask(alpha, thickness=ring_thickness)
    if not ring.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    # Parche del fondo bajo el objeto, desenfocado localmente.
    patch = canvas_rgb[y1:y2, x1:x2].astype(np.float32)
    patch_blur = cv2.GaussianBlur(patch, (0, 0), sigmaX=bg_blur_sigma, sigmaY=bg_blur_sigma)

    rgb = out[..., :3].astype(np.float32)
    ring_n = (ring.astype(np.float32) / 255.0) * float(mix)
    weight = ring_n[sy1:sy2, sx1:sx2, None]
    rgb_obj_crop = rgb[sy1:sy2, sx1:sx2]
    rgb_obj_crop[:] = rgb_obj_crop * (1.0 - weight) + patch_blur * weight

    rgb[sy1:sy2, sx1:sx2] = rgb_obj_crop
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out
