"""
Limpieza del borde alfa: descontaminación de halo, micro-suavizado y fusión de la costura.

Los píxeles del interior sólido del producto NO se tocan. Solo trabajamos sobre el
anillo de alfa parcial y sobre el propio canal alfa.
"""

import cv2
import numpy as np


def clean_alpha_edges(rgba: np.ndarray) -> np.ndarray:
    """
    Versión migrada de utils.pre_procesar_objeto_universal. Hace tres cosas:
    inpainting Telea sobre las zonas transparentes para limpiar bordes RGB sucios,
    una erosión elíptica de 1 px que respeta las curvas, y un micro-suavizado
    gaussiano (sigma ~0.8) del alfa para conseguir bordes blandos.

    Trabaja sobre una copia y conserva el RGB del interior sólido de la silueta.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("clean_alpha_edges espera un RGBA HxWx4")

    rgb = rgba[..., :3].copy()
    alpha = rgba[..., 3].copy()

    # El inpainting Telea solo cubre los píxeles muy transparentes (< 200 de alfa),
    # así que el RGB del interior sólido nunca se reescribe.
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
    Elimina el halo de color residual: en los píxeles con 0 < alfa < 255 (el anillo
    de la costura) sustituye el RGB por el promedio del RGB de los vecinos totalmente
    opacos dentro de una ventana pequeña.

    Usa cv2.boxFilter restringido a los píxeles opacos mediante una acumulación con máscara.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("decontaminate_alpha_edges espera un RGBA HxWx4")

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3]

    solid = (alpha >= 250).astype(np.float32)            # pesos
    rgb_w = rgb * solid[..., None]                        # cero donde no es sólido

    k = 2 * ring_px + 1
    rgb_sum = cv2.boxFilter(rgb_w, ddepth=-1, ksize=(k, k), normalize=False)
    w_sum   = cv2.boxFilter(solid, ddepth=-1, ksize=(k, k), normalize=False)
    safe    = np.maximum(w_sum, 1e-3)[..., None]
    rgb_mean_neighbours = rgb_sum / safe

    seam = (alpha > 0) & (alpha < 250)
    out_rgb = rgb.copy()
    out_rgb[seam] = rgb_mean_neighbours[seam]

    # Si dentro de la ventana no hay ningún vecino sólido, dejamos el RGB original.
    no_neighbour = (w_sum < 0.5)
    out_rgb[no_neighbour] = rgb[no_neighbour]

    out = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), alpha])
    return out


def feather_alpha(rgba: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Aplica un desenfoque gaussiano solo al canal alfa. El RGB queda intacto."""
    if sigma <= 0:
        return rgba
    out = rgba.copy()
    out[..., 3] = cv2.GaussianBlur(out[..., 3], (0, 0), sigmaX=sigma, sigmaY=sigma)
    return out


def edge_ring_mask(alpha: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Devuelve una máscara uint8 HxW (0..255) que marca el anillo de la costura alrededor
    del objeto: dilate(alfa) - erode(alfa). Sirve para la fusión atmosférica.
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
    Limpieza estándar previa a la composición: primero se descontamina el halo (usando el
    RGB sólido original) y luego se suaviza ligeramente el alfa. Aquí no se ejecuta el
    inpainting Telea a propósito: la descontaminación elimina el halo de forma más selectiva.
    """
    rgba = decontaminate_alpha_edges(rgba, ring_px=3)
    rgba = feather_alpha(rgba, sigma=0.8)
    return rgba
