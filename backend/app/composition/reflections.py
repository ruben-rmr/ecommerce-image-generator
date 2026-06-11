"""
Reflejo vertical opcional para suelos brillantes/reflectantes (mármol, agua).
Es un simple espejado de píxeles con un degradado de alfa vertical. Sin raytracing ni IA.
"""

import cv2
import numpy as np


def make_reflection(rgba: np.ndarray,
                    fade: float = 0.35,
                    blur_sigma: float = 1.0) -> np.ndarray:
    """
    Crea una copia volteada y con el alfa atenuado del objeto, lista para pegarse justo
    debajo de él en el lienzo. Devuelve un RGBA uint8 HxWx4 del mismo tamaño que la entrada.

    `fade` es el alfa en la parte alta del reflejo (la más cercana al objeto), que decae
    linealmente hasta 0 en la parte baja. `blur_sigma` simula la micro-rugosidad del suelo
    (pequeño para suelos pulidos, mayor para suelos mate).
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("make_reflection espera un RGBA HxWx4")

    flipped = cv2.flip(rgba, 0)  # volteo vertical
    rgb = flipped[..., :3].astype(np.float32)
    alpha = flipped[..., 3].astype(np.float32)

    h, _ = alpha.shape
    fade_curve = np.linspace(fade, 0.0, h, dtype=np.float32).reshape(-1, 1)
    alpha = alpha * fade_curve

    if blur_sigma > 0.05:
        # Domina el desenfoque horizontal para un brillo de suelo más creíble.
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
    Calcula dónde pegar el reflejo para que su borde superior quede a la altura de los pies
    del objeto. `feet_y_local` es la y del píxel más bajo del objeto en coordenadas LOCALES
    del objeto.
    """
    x, y = obj_top_left
    ow, oh = obj_size
    # Tras el volteo, los "pies" del objeto quedan arriba; los alineamos en y + feet_y_local.
    return (x, y + feet_y_local + 1)
