"""
Alpha-edge cleanup: halo decontamination, micro-feathering, seam blending.

The product's solid interior pixels are left UNTOUCHED. We only operate on
the partial-alpha annulus and the alpha channel itself.
"""

import cv2
import numpy as np


def clean_alpha_edges(rgba: np.ndarray) -> np.ndarray:
    """
    Migration of utils.pre_procesar_objeto_universal:
      1. Telea inpainting on transparent regions to clean dirty RGB borders.
      2. Elliptical erosion (1px) to respect curves.
      3. Gaussian micro-feather (sigma ~0.8) on alpha for soft edges.

    Operates on a copy. RGB inside the solid silhouette is preserved.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("clean_alpha_edges expects HxWx4 RGBA")

    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3].copy()

    # Telea inpainting only over very-transparent pixels (< 200 alpha) so the
    # solid interior RGB is never rewritten.
    _, mask_solid = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    mask_to_fill = cv2.bitwise_not(mask_solid)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_clean = cv2.inpaint(bgr, mask_to_fill, 5, cv2.INPAINT_TELEA)
    rgb_clean = cv2.cvtColor(bgr_clean, cv2.COLOR_BGR2RGB)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_eroded = cv2.erode(alpha, kernel, iterations=1)
    alpha_soft = cv2.GaussianBlur(alpha_eroded, (0, 0), sigmaX=0.8, sigmaY=0.8)

    out = np.dstack([rgb_clean, alpha_soft])
    return out


def decontaminate_alpha_edges(rgba: np.ndarray, ring_px: int = 3) -> np.ndarray:
    """
    Remove residual color halo: in pixels with 0 < alpha < 255 (the seam ring),
    replace RGB by the mean RGB of fully-opaque neighbours within a small window.

    Uses cv2.blur restricted to fully-opaque pixels via masked accumulation.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("decontaminate_alpha_edges expects HxWx4 RGBA")

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3]

    solid = (alpha >= 250).astype(np.float32)            # weights
    rgb_w = rgb * solid[..., None]                       # zero where not solid

    k = 2 * ring_px + 1
    rgb_sum = cv2.boxFilter(rgb_w, ddepth=-1, ksize=(k, k), normalize=False)
    w_sum   = cv2.boxFilter(solid, ddepth=-1, ksize=(k, k), normalize=False)
    safe    = np.maximum(w_sum, 1e-3)[..., None]
    rgb_mean_neighbours = rgb_sum / safe

    seam = (alpha > 0) & (alpha < 250)
    out_rgb = rgb.copy()
    out_rgb[seam] = rgb_mean_neighbours[seam]

    # Where there are NO solid neighbours within the window, keep original RGB.
    no_neighbour = (w_sum < 0.5)
    out_rgb[no_neighbour] = rgb[no_neighbour]

    out = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), alpha])
    return out


def feather_alpha(rgba: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Gaussian blur the alpha channel only. Keeps RGB intact."""
    if sigma <= 0:
        return rgba
    out = rgba.copy()
    out[..., 3] = cv2.GaussianBlur(out[..., 3], (0, 0), sigmaX=sigma, sigmaY=sigma)
    return out


def edge_ring_mask(alpha: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Returns a uint8 HxW mask (0..255) marking the 'seam' annulus around the
    object: dilate(alpha) - erode(alpha). Useful for atmospheric blending.
    """
    binary = (alpha > 8).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness + 1, 2 * thickness + 1))
    dil = cv2.dilate(binary, k, iterations=1)
    ero = cv2.erode(binary, k, iterations=1)
    ring = cv2.subtract(dil, ero)
    ring = cv2.GaussianBlur(ring, (0, 0), sigmaX=thickness * 0.6, sigmaY=thickness * 0.6)
    return ring


def prepare_object(rgba: np.ndarray) -> np.ndarray:
    """
    Standard pre-composition cleanup: decontaminate halo first (uses original
    solid RGB), then feather alpha lightly. Telea inpainting is intentionally
    NOT run here — decontamination is a more targeted halo killer.
    """
    rgba = decontaminate_alpha_edges(rgba, ring_px=3)
    rgba = feather_alpha(rgba, sigma=0.8)
    return rgba
