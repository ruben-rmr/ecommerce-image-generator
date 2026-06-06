"""
Figura 4.18 — Descomposición de la sombra proyectada.

Genera las tres etapas internas de `composition.shadows.cast_shadow`, que la
función no expone porque solo devuelve la máscara final:

    1. Silueta original          -> binary (alpha umbralizado)
    2. Escorzo + cizallamiento    -> warpAffine(binary) ANTES del desenfoque
    3. Penumbra graduada + fade   -> resultado final (dos blurs + rampa tip + fade)

Cada etapa se renderiza sobre un lienzo blanco para que se lea bien en la memoria.
Se guardan los tres paneles por separado y un montaje horizontal con rótulos.

Requiere un PNG de producto CON FONDO TRANSPARENTE (RGBA): la sombra se calcula
a partir del canal alfa. Un PNG con fondo gris/opaco produciría una sombra
rectangular (todo el lienzo sería "silueta").

Uso:
    python tools/figura_sombra.py --input tools/_input/producto.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

# --- Hacer importable el paquete `composition` (vive en backend/app) ----------
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app"))
sys.path.insert(0, APP_DIR)

from composition.io_utils import png_bytes_to_rgba, multiply_shadow, alpha_blend  # noqa: E402
from composition.placement import auto_scale  # noqa: E402
from composition.edges import prepare_object  # noqa: E402
from composition.shadows import _alpha_binary, _vertical_fade_ramp  # noqa: E402

# Mismos parámetros que usa el modo estudio (studio.py) para la sombra proyectada.
LIGHT_DIR = (-0.6, -0.6)
LENGTH = 0.45
SQUASH = 0.20
FADE = 0.5
SIGMA_CONTACT = 4.0
SIGMA_TIP = 20.0
INTENSITY = 0.55


def _shadow_stages(alpha, top_left, canvas_size):
    """Reproduce cast_shadow exponiendo los buffers intermedios (en coords de lienzo)."""
    W, H = canvas_size
    oh, ow = alpha.shape[:2]
    x, y = top_left

    binary = _alpha_binary(alpha).astype(np.float32) / 255.0
    ys, _ = np.where(alpha > 16)
    y_anchor = int(ys.max()) if ys.size else oh - 1

    # Etapa 1: silueta erguida, solo trasladada al lienzo.
    M_id = np.float32([[1, 0, x], [0, 1, y]])
    sil_full = cv2.warpAffine(binary, M_id, (W, H), flags=cv2.INTER_NEAREST, borderValue=0.0)

    # Matriz afín: escorzo vertical (squash) + cizallamiento opuesto a la luz.
    lx, ly = LIGHT_DIR
    norm = max(np.hypot(lx, ly), 1e-6)
    lx, ly = lx / norm, ly / norm
    shear_x = -lx * LENGTH
    M = np.float32([
        [1.0, shear_x, x - shear_x * y_anchor],
        [0.0, SQUASH, y + y_anchor * (1.0 - SQUASH)],
    ])

    # Etapa 2: geometría pura del warp (sin fade, sin blur).
    warped_geom = cv2.warpAffine(binary, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)

    # Etapa 3: fade base->punta + penumbra graduada (nítida en contacto, difusa en punta).
    field = binary * _vertical_fade_ramp((oh, ow), FADE)
    tip = np.repeat(np.linspace(1.0, 0.0, oh, dtype=np.float32).reshape(-1, 1), ow, axis=1)
    warped = cv2.warpAffine(field, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)
    warped_tip = cv2.warpAffine(tip, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0.0)
    sharp = cv2.GaussianBlur(warped, (0, 0), sigmaX=SIGMA_CONTACT, sigmaY=SIGMA_CONTACT)
    soft = cv2.GaussianBlur(warped, (0, 0), sigmaX=SIGMA_TIP, sigmaY=SIGMA_TIP)
    w = np.clip(warped_tip, 0.0, 1.0)
    result = (sharp * (1.0 - w) + soft * w) * INTENSITY

    return sil_full, warped_geom, result


def _paste_object(canvas_rgb, obj_rgba, top_left):
    H, W = canvas_rgb.shape[:2]
    oh, ow = obj_rgba.shape[:2]
    x, y = top_left
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + ow), min(H, y + oh)
    if x2 <= x1 or y2 <= y1:
        return canvas_rgb
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    crop = obj_rgba[sy1:sy2, sx1:sx2]
    out = canvas_rgb.copy()
    out[y1:y2, x1:x2] = alpha_blend(crop[..., :3], crop[..., 3], canvas_rgb[y1:y2, x1:x2])
    return out


def _label(panel, text):
    """Rótulo blanco sobre banda oscura en la parte inferior (ASCII, sin tildes)."""
    h, w = panel.shape[:2]
    band = panel.copy()
    cv2.rectangle(band, (0, h - 34), (w, h), (35, 35, 35), -1)
    out = cv2.addWeighted(band, 0.85, panel, 0.15, 0)
    cv2.putText(out, text, (12, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Genera la Figura 4.18 (descomposicion de la sombra).")
    ap.add_argument("--input", default=os.path.join("tools", "_input", "producto.png"),
                    help="PNG de producto con fondo TRANSPARENTE (RGBA).")
    ap.add_argument("--outdir", default=os.path.join("tools", "figuras"),
                    help="Carpeta de salida.")
    ap.add_argument("--canvas", type=int, default=900, help="Lado del lienzo cuadrado.")
    ap.add_argument("--height-ratio", type=float, default=0.50,
                    help="Altura de la silueta respecto al lienzo.")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"❌ No existe el fichero de entrada: {args.input}\n"
                 f"   Coloca ahi el PNG transparente del producto (salida de 'Segmentar').")

    with open(args.input, "rb") as f:
        rgba = png_bytes_to_rgba(f.read())

    # Validar que realmente hay transparencia.
    if int(rgba[..., 3].min()) >= 250:
        sys.exit("❌ El PNG no tiene transparencia (alfa opaco en toda la imagen).\n"
                 "   Necesito el recorte RGBA, no la imagen con fondo gris.")

    os.makedirs(args.outdir, exist_ok=True)
    canvas_size = (args.canvas, args.canvas)

    # Limpieza de bordes + escalado/colocacion identicos al pipeline real.
    rgba = prepare_object(rgba)
    obj, top_left = auto_scale(rgba, canvas_size, target_height_ratio=args.height_ratio)
    alpha = obj[..., 3]

    sil, warped_geom, shadow = _shadow_stages(alpha, top_left, canvas_size)

    white = np.full((args.canvas, args.canvas, 3), 250, dtype=np.uint8)

    # Panel 1: silueta erguida (gris oscuro).
    p1 = multiply_shadow(white, sil, color=(60, 60, 60), opacity=0.92)
    # Panel 2: silueta proyectada (escorzo + cizallamiento), gris oscuro.
    p2 = multiply_shadow(white, warped_geom, color=(60, 60, 60), opacity=0.92)
    # Panel 3: sombra final difusa + objeto colocado encima.
    p3 = multiply_shadow(white, shadow, color=(0, 0, 0), opacity=1.0)
    p3 = _paste_object(p3, obj, top_left)

    cv2.imwrite(os.path.join(args.outdir, "fig418_1_silueta.png"), cv2.cvtColor(p1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "fig418_2_escorzo.png"), cv2.cvtColor(p2, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.outdir, "fig418_3_penumbra.png"), cv2.cvtColor(p3, cv2.COLOR_RGB2BGR))

    # Montaje horizontal con rotulos.
    gut = 16
    p1l = _label(p1, "1. Silueta original")
    p2l = _label(p2, "2. Escorzo + cizallamiento")
    p3l = _label(p3, "3. Penumbra graduada")
    strip = np.full((args.canvas, args.canvas * 3 + gut * 2, 3), 255, dtype=np.uint8)
    strip[:, 0:args.canvas] = p1l
    strip[:, args.canvas + gut:args.canvas * 2 + gut] = p2l
    strip[:, args.canvas * 2 + gut * 2:] = p3l
    montage = os.path.join(args.outdir, "fig418_montaje.png")
    cv2.imwrite(montage, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    print("✅ Figura 4.18 generada:")
    for n in ("fig418_1_silueta.png", "fig418_2_escorzo.png", "fig418_3_penumbra.png", "fig418_montaje.png"):
        print(f"   - {os.path.join(args.outdir, n)}")


if __name__ == "__main__":
    main()