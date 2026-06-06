"""
Figura 4.19 — Efecto de la armonización LAB.

Compone el MISMO producto sobre el MISMO fondo dos veces, variando únicamente
`harmonize_strength` (la intensidad de la transferencia de color LAB Reinhard de
`composition.harmonization.harmonize_lab`):

    - izquierda: harmonize_strength = 0.0   (sin armonización)
    - derecha:   harmonize_strength = 0.85  (con armonización)

Todo lo demás (colocación, sombra, relight, fusión atmosférica) es idéntico en
ambas, por lo que la diferencia visible aísla el efecto de la armonización LAB.

Requiere un PNG de producto CON FONDO TRANSPARENTE (RGBA) y un fondo del catálogo.
Los metadatos (ground_y, light_dir, reflective…) se leen del sidecar .json del
fondo si existe, igual que hace el catálogo en producción.

Uso:
    python tools/figura_armonizacion.py ^
        --input tools/_input/producto.png ^
        --bg backend/app/backgrounds/oceano/escena_1.png
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# --- Hacer importable el paquete `composition` (vive en backend/app) ----------
APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "app"))
sys.path.insert(0, APP_DIR)

from composition import compose_scene  # noqa: E402
from composition.io_utils import png_bytes_to_rgba  # noqa: E402

STRENGTH_OFF = 0.0
STRENGTH_ON = 0.15


def _load_metadata(bg_path):
    """Lee el sidecar <fondo>.json si existe (mismo esquema que el catálogo)."""
    meta_path = os.path.splitext(bg_path)[0] + ".json"
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = {}
    for k in ("ground_y", "light_dir", "reflective", "reflective_type"):
        if k in raw and raw[k] is not None:
            meta[k] = raw[k]
    return meta


def _label(panel, text):
    """Rótulo blanco sobre banda oscura (ASCII, sin tildes)."""
    h, w = panel.shape[:2]
    out = panel.copy()
    band = out.copy()
    cv2.rectangle(band, (0, 0), (w, 38), (35, 35, 35), -1)
    out = cv2.addWeighted(band, 0.85, out, 0.15, 0)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="Genera la Figura 4.19 (con/sin armonización LAB).")
    ap.add_argument("--input", default=os.path.join("tools", "_input", "producto.png"),
                    help="PNG de producto con fondo TRANSPARENTE (RGBA).")
    ap.add_argument("--bg", default=os.path.join("backend", "app", "backgrounds", "oceano", "escena_1.png"),
                    help="Fondo del catálogo (JPG/PNG).")
    ap.add_argument("--outdir", default=os.path.join("tools", "figuras"),
                    help="Carpeta de salida.")
    ap.add_argument("--strength", type=float, default=STRENGTH_ON,
                    help="Intensidad de armonización del panel 'con' (0..1).")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"❌ No existe el fichero de entrada: {args.input}\n"
                 f"   Coloca ahi el PNG transparente del producto (salida de 'Segmentar').")
    if not os.path.isfile(args.bg):
        sys.exit(f"❌ No existe el fondo: {args.bg}")

    with open(args.input, "rb") as f:
        png_bytes = f.read()

    rgba = png_bytes_to_rgba(png_bytes)
    if int(rgba[..., 3].min()) >= 250:
        sys.exit("❌ El PNG no tiene transparencia (alfa opaco en toda la imagen).\n"
                 "   Necesito el recorte RGBA, no la imagen con fondo gris.")

    os.makedirs(args.outdir, exist_ok=True)
    metadata = _load_metadata(args.bg)
    print(f"ℹ️  Metadatos del fondo: {metadata or '(ninguno)'}")

    off_bytes = compose_scene(png_bytes, args.bg, metadata, harmonize_strength=STRENGTH_OFF)
    on_bytes = compose_scene(png_bytes, args.bg, metadata, harmonize_strength=args.strength)

    off = cv2.imdecode(np.frombuffer(off_bytes, np.uint8), cv2.IMREAD_COLOR)  # BGR
    on = cv2.imdecode(np.frombuffer(on_bytes, np.uint8), cv2.IMREAD_COLOR)

    cv2.imwrite(os.path.join(args.outdir, "fig419_sin_armonizacion.png"), off)
    cv2.imwrite(os.path.join(args.outdir, "fig419_con_armonizacion.png"), on)

    # Montaje lado a lado con rótulos.
    h = min(off.shape[0], on.shape[0])
    off_r = cv2.resize(off, (int(off.shape[1] * h / off.shape[0]), h))
    on_r = cv2.resize(on, (int(on.shape[1] * h / on.shape[0]), h))
    off_lbl = _label(off_r, "Sin armonizacion (0%)")
    on_lbl = _label(on_r, f"Con armonizacion ({int(args.strength * 100)}%)")
    gut = 16
    strip = np.full((h, off_lbl.shape[1] + on_lbl.shape[1] + gut, 3), 255, dtype=np.uint8)
    strip[:, 0:off_lbl.shape[1]] = off_lbl
    strip[:, off_lbl.shape[1] + gut:] = on_lbl
    cv2.imwrite(os.path.join(args.outdir, "fig419_montaje.png"), strip)

    print("✅ Figura 4.19 generada:")
    for n in ("fig419_sin_armonizacion.png", "fig419_con_armonizacion.png", "fig419_montaje.png"):
        print(f"   - {os.path.join(args.outdir, n)}")


if __name__ == "__main__":
    main()