"""Funciones de E/S y utilidades de espacio de color compartidas por el pipeline de composición."""

import io
import numpy as np
import cv2
from PIL import Image


def png_bytes_to_rgba(data: bytes) -> np.ndarray:
    """Decodifica unos bytes PNG a un array RGBA uint8 de HxWx4."""
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.array(img)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Codifica un array RGBA uint8 (HxWx4) a bytes PNG."""
    pil = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Codifica un array RGB uint8 (HxWx3) a bytes PNG."""
    pil = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def load_background(path: str, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Carga un fondo JPG/PNG como RGB (HxWx3, uint8), respetando la orientación EXIF.
    Si se indica target_size=(W, H), la imagen se redimensiona a ese tamaño.
    """
    pil = Image.open(path)
    try:
        from PIL import ImageOps
        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if target_size is not None:
        pil = pil.resize(target_size, Image.LANCZOS)
    return np.array(pil)


def alpha_blend(fg_rgb: np.ndarray, fg_a: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """
    Mezcla alfa estándar (over). El alfa fg_a viene en [0, 255]. Devuelve RGB uint8.
    Las tres entradas deben compartir el mismo tamaño HxW.
    """
    a = fg_a.astype(np.float32) / 255.0
    a3 = a[..., None]
    out = fg_rgb.astype(np.float32) * a3 + bg_rgb.astype(np.float32) * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


def multiply_shadow(bg_rgb: np.ndarray, shadow_alpha: np.ndarray, color=(0, 0, 0), opacity: float = 1.0) -> np.ndarray:
    """
    Oscurece bg_rgb con una sombra suave (HxW, float [0..1] o uint8). El color de la
    sombra es negro por defecto; opacity escala el efecto final. Devuelve RGB uint8.
    """
    if shadow_alpha.dtype != np.float32:
        s = shadow_alpha.astype(np.float32) / 255.0
    else:
        s = shadow_alpha
    s = np.clip(s * float(opacity), 0.0, 1.0)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = bg_rgb.astype(np.float32) * (1.0 - s) + color_arr * s
    return np.clip(out, 0, 255).astype(np.uint8)


def paste_rgba_onto_canvas(canvas_rgb: np.ndarray, obj_rgba: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    """
    Pega obj_rgba (HxWx4) sobre canvas_rgb (HxWx3) anclando su esquina superior
    izquierda en (x, y). Recorta a los límites del lienzo. Devuelve una copia RGB uint8.
    """
    H, W = canvas_rgb.shape[:2]
    oh, ow = obj_rgba.shape[:2]
    x, y = top_left

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return canvas_rgb.copy()

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    roi = canvas_rgb[y1:y2, x1:x2].copy()
    obj_crop = obj_rgba[sy1:sy2, sx1:sx2]
    blended = alpha_blend(obj_crop[..., :3], obj_crop[..., 3], roi)

    out = canvas_rgb.copy()
    out[y1:y2, x1:x2] = blended
    return out


def compute_object_footprint(alpha: np.ndarray, threshold: int = 32) -> tuple[int, int, int, int]:
    """
    Devuelve el bbox ajustado (x_min, y_min, x_max, y_max) de la silueta según su alfa.
    Si ningún píxel supera el umbral, devuelve la imagen completa.
    """
    mask = alpha > threshold
    if not mask.any():
        h, w = alpha.shape[:2]
        return 0, 0, w - 1, h - 1
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def resize_rgba(rgba: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """Redimensiona un array RGBA (HxWx4) conservando el alfa (LANCZOS de cv2)."""
    return cv2.resize(rgba, new_size, interpolation=cv2.INTER_LANCZOS4)


def print_stage_timings(title: str,
                        timings: list[tuple[str, float]],
                        total_ms: float,
                        total_label: str = "Total") -> None:
    """
    Imprime el desglose de tiempos por etapa (en ms) como una tabla alineada, útil para
    perfilar el pipeline de composición. `timings` es una lista ordenada de pares
    (nombre_de_etapa, milisegundos).
    """
    label_w = max([len(name) for name, _ in timings] + [len(total_label)])
    print(f"\n  {title}")
    for name, ms in timings:
        pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
        print(f"   {name:<{label_w}} : {ms:8.2f} ms  ({pct:5.1f} %)")
    print(f"   {'-' * (label_w + 25)}")
    print(f"   {total_label:<{label_w}} : {total_ms:8.2f} ms  (100.0 %)")


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> LAB float32 (escala de OpenCV: L 0-100, a/b ~-128..127)."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """LAB float32 -> RGB uint8."""
    lab8 = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB)
