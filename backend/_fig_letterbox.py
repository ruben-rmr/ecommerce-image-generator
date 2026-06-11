"""
Genera la figura didáctica del letterbox: tres viñetas anotadas que muestran
cómo "preprocess_image()" transforma la imagen y remapea el bbox.

  (a) imagen original con su bbox
  (b) imagen escalada por el lado mayor (sin padding), bbox escalado
  (c) lienzo cuadrado con relleno gris 114 y el bbox ya remapeado

Anota scale, pad_x y pad_y — los valores que guarda "meta" y que luego usa
postprocess_mask() para revertir el preprocesado.

Ejecutar desde backend/ (mismos imports planos que el resto de scripts sueltos):

    .\venv\Scripts\activate
    cd backend
    python _fig_letterbox.py <imagen> [x1 y1 x2 y2]

El bbox es opcional: en coords absolutas (px) o relativas (0-1). Si se omite,
se usa un bbox de ejemplo centrado al 60% de la imagen.
Salida: _fig_letterbox.png en el directorio actual.
"""
import sys
import cv2
import numpy as np
from PIL import Image

from app.segmentation import preprocess_image, _bbox_to_canvas

GREEN = (0, 200, 0)
GRAY_PAD = 114
PANEL_BG = (255, 255, 255)
TEXT = (30, 30, 30)
ARROW = (0, 110, 220)


def _parse_bbox(args, w, h):
    """Devuelve bbox [x1,y1,x2,y2] en px absolutos a partir de los argumentos."""
    if len(args) == 4:
        vals = [float(v) for v in args]
        if all(0.0 <= v <= 1.0 for v in vals):  # relativo 0-1
            x1, y1, x2, y2 = vals
            return [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
        return [int(v) for v in vals]
    # Por defecto: caja centrada al 60%
    return [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)]


def _draw_bbox(img, bbox, thickness):
    out = img.copy()
    cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), GREEN, thickness)
    return out


def _label(img, text, color=TEXT, scale=0.7, thickness=2):
    """Añade una barra de título encima de un panel."""
    bar_h = 34
    h, w = img.shape[:2]
    out = np.full((h + bar_h, w, 3), PANEL_BG, np.uint8)
    out[bar_h:, :] = img
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return out


def _pad_to_height(img, target_h):
    """Centra verticalmente un panel sobre fondo blanco hasta target_h."""
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    out = np.full((target_h, w, 3), PANEL_BG, np.uint8)
    top = (target_h - h) // 2
    out[top:top + h, :] = img
    return out


def build_figure(image_path, bbox_args):
    pil = Image.open(image_path).convert("RGB")
    img_np = np.array(pil)
    h, w = img_np.shape[:2]
    bbox = _parse_bbox(bbox_args, w, h)

    # Pipeline real: una sola fuente de verdad para los valores.
    canvas, meta = preprocess_image(img_np, target_size=1024)
    scale = meta["scale"]
    pad_x, pad_y = meta["pad_x"], meta["pad_y"]
    new_w, new_h = meta["new_w"], meta["new_h"]
    cb = _bbox_to_canvas(bbox, meta)

    # (a) Original con bbox
    th_a = max(2, w // 300)
    panel_a = _draw_bbox(img_np, bbox, th_a)

    # (b) Escalada por el lado mayor (la región útil del canvas, sin padding)
    resized = canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w].copy()
    # bbox escalado al espacio "resized" = bbox del canvas menos el padding
    rb = [cb[0] - pad_x, cb[1] - pad_y, cb[2] - pad_x, cb[3] - pad_y]
    panel_b = _draw_bbox(resized, rb, max(2, new_w // 300))

    # (c) Canvas cuadrado con padding gris y bbox remapeado
    panel_c = _draw_bbox(canvas, cb, 2)
    # Marcar el padding con líneas guía
    cv2.line(panel_c, (0, pad_y), (meta["canvas_size"], pad_y), ARROW, 1, cv2.LINE_AA)
    cv2.line(panel_c, (0, pad_y + new_h), (meta["canvas_size"], pad_y + new_h), ARROW, 1, cv2.LINE_AA)
    cv2.line(panel_c, (pad_x, 0), (pad_x, meta["canvas_size"]), ARROW, 1, cv2.LINE_AA)
    cv2.line(panel_c, (pad_x + new_w, 0), (pad_x + new_w, meta["canvas_size"]), ARROW, 1, cv2.LINE_AA)

    # Normalizamos a una altura común para alinear las viñetas.
    target = 460
    def _fit(p):
        ph, pw = p.shape[:2]
        s = target / ph
        return cv2.resize(p, (int(pw * s), target), interpolation=cv2.INTER_AREA)

    panel_a = _label(_fit(panel_a), f"(a) original  {w}x{h}")
    panel_b = _label(_fit(panel_b), f"(b) escalada  {new_w}x{new_h}  scale={scale:.3f}")
    panel_c = _label(_fit(panel_c), f"(c) canvas {meta['canvas_size']}^2  pad=({pad_x},{pad_y})")

    # Montaje horizontal con separadores.
    H = max(p.shape[0] for p in (panel_a, panel_b, panel_c))
    panels = [_pad_to_height(p, H) for p in (panel_a, panel_b, panel_c)]
    gap = np.full((H, 24, 3), PANEL_BG, np.uint8)
    arrow_col = gap.copy()
    cv2.arrowedLine(arrow_col, (3, H // 2), (20, H // 2), ARROW, 2, tipLength=0.5)

    fig = np.hstack([panels[0], arrow_col, panels[1], arrow_col, panels[2]])

    # Pie con los valores de meta.
    footer_h = 40
    fig_full = np.full((fig.shape[0] + footer_h, fig.shape[1], 3), PANEL_BG, np.uint8)
    fig_full[:fig.shape[0], :] = fig
    footer = (f"meta: scale={scale:.4f}  pad_x={pad_x}  pad_y={pad_y}  "
              f"new=({new_w}x{new_h})  orig=({w}x{h})  -> postprocess_mask() revierte esto")
    cv2.putText(fig_full, footer, (8, fig.shape[0] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)

    out_path = "_fig_letterbox.png"
    cv2.imwrite(out_path, cv2.cvtColor(fig_full, cv2.COLOR_RGB2BGR))
    print(f"OK -> {out_path}  ({fig_full.shape[1]}x{fig_full.shape[0]})")
    print(f"   scale={scale:.4f}  pad=({pad_x},{pad_y})  "
          f"new=({new_w}x{new_h})  bbox_canvas={cb}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: python _fig_letterbox.py <imagen> [x1 y1 x2 y2]")
        sys.exit(1)
    build_figure(sys.argv[1], sys.argv[2:])
