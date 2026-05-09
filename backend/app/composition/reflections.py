"""
Optional vertical reflection for shiny / reflective floors (marble, water).
Pure pixel mirror with a vertical alpha fade. No raytracing, no IA.
"""

import cv2
import numpy as np


def make_reflection(rgba: np.ndarray,
                    fade: float = 0.35,
                    blur_sigma: float = 1.0) -> np.ndarray:
    """
    Build a flipped, alpha-faded copy of the object suitable for placing
    directly under it on the canvas. Returns HxWx4 uint8 RGBA, same size as
    input.

    `fade` is the alpha at the top of the reflection (closest to the object);
    decays linearly to 0 at the bottom. `blur_sigma` simulates micro-roughness
    on the floor (tiny for glossy, larger for matte).
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("make_reflection expects HxWx4 RGBA")

    flipped = cv2.flip(rgba, 0)  # vertical flip
    rgb = flipped[..., :3].astype(np.float32)
    alpha = flipped[..., 3].astype(np.float32)

    h, _ = alpha.shape
    fade_curve = np.linspace(fade, 0.0, h, dtype=np.float32).reshape(-1, 1)
    alpha = alpha * fade_curve

    if blur_sigma > 0.05:
        # Slight horizontal blur dominates for a believable floor sheen.
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=blur_sigma * 4.0, sigmaY=blur_sigma)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=blur_sigma * 4.0, sigmaY=blur_sigma)

    out = np.dstack([
        np.clip(rgb, 0, 255).astype(np.uint8),
        np.clip(alpha, 0, 255).astype(np.uint8),
    ])
    return out


def reflection_top_left(obj_top_left: tuple[int, int], obj_size: tuple[int, int],
                        feet_y_local: int) -> tuple[int, int]:
    """
    Compute where to paste the reflection so its top edge sits at the object's
    feet (`feet_y_local` is the y of the lowest object pixel in OBJECT-local
    coordinates).
    """
    x, y = obj_top_left
    ow, oh = obj_size
    # The flipped image's "feet" are at top; we want them aligned at y + feet_y_local.
    return (x, y + feet_y_local + 1)
