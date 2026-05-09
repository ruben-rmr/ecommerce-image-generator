"""
Shadow synthesis: contact, drop, fake ambient occlusion, scene-directional shadow.

All shadows are rendered as a single-channel "shadow alpha" (HxW float32 in [0..1])
that the caller multiplies onto the canvas with a chosen color/opacity.
"""

import cv2
import numpy as np


def _alpha_binary(alpha: np.ndarray, threshold: int = 16) -> np.ndarray:
    return (alpha > threshold).astype(np.uint8) * 255


def contact_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                   intensity: float = 0.55, sigma: float = 3.0, band_ratio: float = 0.08) -> np.ndarray:
    """
    Sharp dark contact shadow under the object's footprint.

    canvas_size = (W, H). The shadow is rendered into a full-canvas float32 mask
    so the caller can multiply directly. `band_ratio` limits the shadow to a
    horizontal strip immediately under the object's lowest pixels.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)

    binary = _alpha_binary(alpha)
    if not binary.any():
        return out

    # Erode lightly so the contact shadow is tighter than the silhouette.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, k, iterations=1)
    blurred = cv2.GaussianBlur(eroded, (0, 0), sigmaX=sigma, sigmaY=sigma)

    oh, ow = alpha.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    crop = blurred[sy1:sy2, sx1:sx2].astype(np.float32) / 255.0

    # Restrict to vertical band near the bottom of the object's footprint.
    ys, _ = np.where(alpha > 16)
    if ys.size:
        y_bottom_obj = int(ys.max())
        band_h = max(int(oh * band_ratio), 8)
        # Build a feathered band mask in object-local coordinates.
        band = np.zeros_like(alpha, dtype=np.float32)
        y_band_top = max(0, y_bottom_obj - int(band_h * 0.4))
        y_band_bot = min(oh, y_bottom_obj + band_h)
        band[y_band_top:y_band_bot] = 1.0
        band = cv2.GaussianBlur(band, (0, 0), sigmaX=band_h * 0.4, sigmaY=band_h * 0.4)
        crop = crop * band[sy1:sy2, sx1:sx2]

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], crop * intensity)
    return out


def drop_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                squash: float = 0.18, shear: float = 0.0,
                offset: tuple[int, int] = (0, 0),
                sigma: float = 18.0, intensity: float = 0.30) -> np.ndarray:
    """
    Soft drop shadow obtained by an affine warp of the silhouette (vertical
    squash + optional horizontal shear) and a large gaussian blur.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    if not binary.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left

    # Anchor warp to the bottom of the silhouette so it stays grounded.
    ys, _ = np.where(alpha > 16)
    y_anchor_local = int(ys.max()) if ys.size else oh - 1

    M = np.float32([
        [1.0, shear, -shear * y_anchor_local],
        [0.0, squash, y_anchor_local * (1.0 - squash)],
    ])
    warped = cv2.warpAffine(binary, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Place into canvas with offset.
    dx, dy = offset
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(W, x + dx + ow), min(H, y + dy + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - (x + dx), y1 - (y + dy)
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], warped[sy1:sy2, sx1:sx2] * intensity)
    return out


def fake_ambient_occlusion(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                           radius: int = 30, intensity: float = 0.15) -> np.ndarray:
    """
    Soft darkening immediately around the silhouette, simulating ambient
    occlusion. Built from the inverted distance transform OUTSIDE the silhouette.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha)
    if not binary.any():
        return out

    inv = cv2.bitwise_not(binary)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ao = np.clip(1.0 - dist / float(radius), 0.0, 1.0)
    ao = cv2.GaussianBlur(ao, (0, 0), sigmaX=radius * 0.25, sigmaY=radius * 0.25)

    oh, ow = alpha.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], ao[sy1:sy2, sx1:sx2] * intensity)
    return out


def scene_shadow(canvas_size: tuple[int, int], alpha: np.ndarray, top_left: tuple[int, int],
                 light_dir: tuple[float, float],
                 length: float = 0.55,
                 squash: float = 0.28,
                 sigma: float = 22.0,
                 intensity: float = 0.40) -> np.ndarray:
    """
    Directional drop shadow using the light vector. The shadow is projected
    OPPOSITE to the light direction; squashed vertically. `length` scales the
    horizontal projection length relative to the object height.
    """
    W, H = canvas_size
    out = np.zeros((H, W), dtype=np.float32)
    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    if not binary.any():
        return out

    oh, ow = alpha.shape[:2]
    x, y = top_left
    ys, _ = np.where(alpha > 16)
    y_anchor_local = int(ys.max()) if ys.size else oh - 1

    lx, ly = float(light_dir[0]), float(light_dir[1])
    norm = max(np.hypot(lx, ly), 1e-6)
    lx, ly = lx / norm, ly / norm

    # Shear the shadow toward (-lx) and squash vertically.
    shear_x = -lx * length
    M = np.float32([
        [1.0, shear_x, -shear_x * y_anchor_local],
        [0.0, squash, y_anchor_local * (1.0 - squash)],
    ])
    warped = cv2.warpAffine(binary, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped = cv2.GaussianBlur(warped, (0, 0), sigmaX=sigma, sigmaY=sigma)

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return out
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], warped[sy1:sy2, sx1:sx2] * intensity)
    return out
