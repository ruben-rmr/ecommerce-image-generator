import cv2
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
from datetime import datetime


_DEBUG_DIR = Path(__file__).parent / "debug_auto"


def detectar_bbox_canny(pil_image: Image.Image, padding: float = 0.03) -> list[int]:
    """
    Detecta el bounding box del objeto principal usando Canny edge detection.

    1. Convierte a escala de grises y aplica GaussianBlur para reducir ruido.
    2. Aplica Canny edge detection.
    3. Busca contornos y calcula el bounding box que los engloba.
    4. Añade un margen (padding) relativo al tamaño de la imagen.

    Args:
        pil_image: Imagen PIL (cualquier modo, se convierte a gris internamente).
        padding: Margen relativo (0.03 = 3 % de cada dimensión).
    Returns:
        [x1, y1, x2, y2] en píxeles absolutos.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Desenfoque para reducir ruido antes de Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Umbrales adaptativos basados en la mediana de la imagen
    median_val = np.median(blurred)
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, low, high)

    # Dilatar bordes para cerrar pequeños huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    # Buscar contornos
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: devolver toda la imagen
        _save_debug_images(img_rgb, edges_dilated, [0, 0, w, h])
        return [0, 0, w, h]

    # Unir todos los contornos y obtener el bbox envolvente
    all_points = np.concatenate(contours)
    rx, ry, rw, rh = cv2.boundingRect(all_points)

    # Añadir padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, rx - pad_x)
    y1 = max(0, ry - pad_y)
    x2 = min(w, rx + rw + pad_x)
    y2 = min(h, ry + rh + pad_y)

    print(f"🔍 Canny bbox detectado: [{x1},{y1},{x2},{y2}] sobre {w}×{h}")
    _save_debug_images(img_rgb, edges_dilated, [x1, y1, x2, y2])
    return [x1, y1, x2, y2]


def _save_debug_images(img_rgb: np.ndarray, edges: np.ndarray, bbox: list[int]) -> None:
    """Guarda imágenes de debug del pipeline Auto-segmentar en debug_auto/."""
    try:
        _DEBUG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # hasta segundos

        # 1. Imagen de Canny edges (dilatada)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(_DEBUG_DIR / f"{ts}_canny_edges.jpg"), edges_bgr)

        # 2. Imagen original con bbox dibujada
        x1, y1, x2, y2 = bbox
        vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"bbox [{x1},{y1},{x2},{y2}]"
        cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(_DEBUG_DIR / f"{ts}_bbox_detected.jpg"), vis)

        print(f"💾 Debug guardado en debug_auto/{ts}_*.jpg")
    except Exception as exc:
        print(f"⚠️  No se pudo guardar debug: {exc}")


def pre_procesar_objeto_universal(pil_image_original):
    """
    Limpieza Elíptica + Micro-Feathering.
    Helper original de limpieza de halos. La pipeline de composición usa una
    versión equivalente en composition/edges.py (clean_alpha_edges).
    """
    if pil_image_original.mode != 'RGBA':
        pil_image_original = pil_image_original.convert('RGBA')

    img_np = np.array(pil_image_original)
    alpha = img_np[:, :, 3]
    rgb = img_np[:, :, :3]

    # Inpainting Telea (Quitar bordes blancos o sucios)
    _, mask_solid = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    mask_to_fill = cv2.bitwise_not(mask_solid)
    rgb_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    rgb_clean_cv = cv2.inpaint(rgb_cv, mask_to_fill, 5, cv2.INPAINT_TELEA)
    rgb_clean = cv2.cvtColor(rgb_clean_cv, cv2.COLOR_BGR2RGB)

    # Erosión Elíptica (Respetar curvas)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha_contracted = cv2.erode(alpha, kernel_ellipse, iterations=1)

    # Micro-Feather (Suavidad fotográfica)
    alpha_pil = Image.fromarray(alpha_contracted)
    alpha_soft = alpha_pil.filter(ImageFilter.GaussianBlur(radius=0.8))

    objeto_limpio = Image.merge('RGBA', (
        Image.fromarray(rgb_clean[:, :, 0]),
        Image.fromarray(rgb_clean[:, :, 1]),
        Image.fromarray(rgb_clean[:, :, 2]),
        alpha_soft
    ))
    return objeto_limpio
