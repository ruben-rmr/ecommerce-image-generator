import cv2
import numpy as np
from PIL import Image, ImageFilter


def pre_procesar_objeto_universal(pil_image_original):
    """
    Limpieza Elíptica + Micro-Feathering.
    Garantiza un recorte perfecto sin halos antes de enviar a APIs externas.
    Se usa en el flujo de generación de fondo (endpoint /generate/).
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
