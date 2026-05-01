import io
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import FastSAM

# ─────────────────────────────────────────────────────────────────────────────
# Cargar modelo FastSAM local (una sola vez al importar el módulo)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "FastSAM-s.pt")
_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"⏳ Cargando FastSAM desde {MODEL_PATH} ...")
        _model = FastSAM(MODEL_PATH)
        print("✅ Modelo FastSAM cargado.")
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# 1. PREPROCESADO: resize controlado + padding centrado en canvas cuadrado
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(img_np: np.ndarray, target_size: int = 1024) -> tuple[np.ndarray, dict]:
    """
    Escala la imagen manteniendo aspect ratio y la centra en un canvas
    cuadrado con padding gris (114, 114, 114) — mismo color que YOLO letterbox.

    Returns:
        canvas: imagen cuadrada (target_size x target_size x 3), uint8 RGB.
        meta: dict con scale, pad_x, pad_y, orig_h, orig_w para revertir.
    """
    h, w = img_np.shape[:2]

    # Escalar por el lado mayor. Evitar upscaling agresivo.
    effective_size = min(target_size, max(h, w))
    scale = effective_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Canvas cuadrado con padding centrado
    canvas = np.full((effective_size, effective_size, 3), 114, dtype=np.uint8)
    pad_x = (effective_size - new_w) // 2
    pad_y = (effective_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_h": h,
        "orig_w": w,
        "canvas_size": effective_size,
        "new_w": new_w,
        "new_h": new_h,
    }
    return canvas, meta


def _bbox_to_canvas(bbox: list[int], meta: dict) -> list[int]:
    """Transforma bbox [x1,y1,x2,y2] de coordenadas originales a coordenadas del canvas."""
    scale, pad_x, pad_y = meta["scale"], meta["pad_x"], meta["pad_y"]
    x1 = int(bbox[0] * scale) + pad_x
    y1 = int(bbox[1] * scale) + pad_y
    x2 = int(bbox[2] * scale) + pad_x
    y2 = int(bbox[3] * scale) + pad_y
    return [x1, y1, x2, y2]


def _point_to_canvas(x: int, y: int, meta: dict) -> tuple[int, int]:
    """Transforma un punto de coordenadas originales a coordenadas del canvas."""
    cx = int(x * meta["scale"]) + meta["pad_x"]
    cy = int(y * meta["scale"]) + meta["pad_y"]
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# 2. INFERENCIA: correr FastSAM sobre el canvas preprocesado
# ─────────────────────────────────────────────────────────────────────────────

def run_fastsam(
    canvas: np.ndarray,
    meta: dict,
    bbox: list[int],
    conf: float = 0.25,
    iou: float = 0.9,
    debug: bool = False,
) -> np.ndarray | None:
    """
    Ejecuta FastSAM sobre el canvas preprocesado y selecciona la mejor máscara.

    Estrategia de prompts:
      1. Point prompt (centro del bbox) — más estable para objetos grandes.
      2. Fallback a box prompt si point no da buen resultado.
      3. Fallback a selección manual por score compuesto.

    Returns:
        Máscara binaria en coordenadas del canvas (canvas_size x canvas_size), o None.
    """
    model = _get_model()
    canvas_size = meta["canvas_size"]

    # Bbox ya transformado al espacio del canvas
    cb = _bbox_to_canvas(bbox, meta)

    # Centro del bbox original → punto en canvas
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    pt_x, pt_y = _point_to_canvas(center_x, center_y, meta)

    print(f"📡 FastSAM inferencia — canvas={canvas_size}×{canvas_size}, "
          f"bbox_canvas={cb}, point=({pt_x},{pt_y})")

    # imgsz = canvas_size porque ya preprocesamos nosotros
    results = model(
        canvas,
        device="cpu",
        retina_masks=True,
        imgsz=canvas_size,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if not results or len(results) == 0 or results[0].masks is None:
        print("⚠️  FastSAM no generó máscaras")
        return None

    if debug:
        _debug_masks(results, cb, canvas_size)

    # ── Intento 1: Point prompt ──────────────────────────────────────────
    mask = _try_point_prompt(canvas, results, pt_x, pt_y, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 2: Box prompt ────────────────────────────────────────────
    mask = _try_box_prompt(canvas, results, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 3: Selección manual por score compuesto ──────────────────
    print("⚠️  Prompts fallaron, selección manual por score")
    return _select_best_mask(results, cb, canvas_size)


def _try_point_prompt(canvas, results, pt_x, pt_y, canvas_bbox, canvas_size):
    """Intenta seleccionar máscara con point prompt (centro del bbox)."""
    try:
        from ultralytics.models.fastsam import FastSAMPrompt
        prompt = FastSAMPrompt(canvas, results, device="cpu")
        ann = prompt.point_prompt(points=[[pt_x, pt_y]], pointlabel=[1])

        mask = _extract_mask(ann, canvas_size)
        if mask is not None:
            # Validar: la máscara debe cubrir al menos 20% del bbox
            if _mask_covers_bbox(mask, canvas_bbox, min_coverage=0.20):
                print("✅ Máscara obtenida vía point_prompt")
                return mask
            else:
                print("⚠️  point_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"⚠️  point_prompt falló: {e}")
    return None


def _try_box_prompt(canvas, results, canvas_bbox, canvas_size):
    """Intenta seleccionar máscara con box prompt."""
    try:
        from ultralytics.models.fastsam import FastSAMPrompt
        prompt = FastSAMPrompt(canvas, results, device="cpu")
        ann = prompt.box_prompt(bbox=canvas_bbox)

        mask = _extract_mask(ann, canvas_size)
        if mask is not None:
            if _mask_covers_bbox(mask, canvas_bbox, min_coverage=0.15):
                print("✅ Máscara obtenida vía box_prompt")
                return mask
            else:
                print("⚠️  box_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"⚠️  box_prompt falló: {e}")
    return None


def _extract_mask(ann, expected_size: int) -> np.ndarray | None:
    """Extrae máscara binaria de la respuesta de FastSAMPrompt."""
    if ann is None:
        return None
    mask_np = ann.cpu().numpy() if hasattr(ann, "cpu") else np.array(ann)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    if mask_np.size == 0 or mask_np.max() == 0:
        return None
    mask_bin = (mask_np > 0.5).astype(np.uint8) * 255
    # Asegurar tamaño correcto (INTER_NEAREST para binario)
    if mask_bin.shape[0] != expected_size or mask_bin.shape[1] != expected_size:
        mask_bin = cv2.resize(mask_bin, (expected_size, expected_size),
                              interpolation=cv2.INTER_NEAREST)
    return mask_bin


def _mask_covers_bbox(mask: np.ndarray, bbox: list[int], min_coverage: float) -> bool:
    """Verifica que la máscara cubre al menos min_coverage del área del bbox."""
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bbox_region = mask[y1:y2, x1:x2]
    if bbox_region.size == 0:
        return False
    coverage = (bbox_region > 0).sum() / bbox_region.size
    return coverage >= min_coverage


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELECCIÓN MANUAL (fallback): score compuesto coverage × overlap × centroid
# ─────────────────────────────────────────────────────────────────────────────

def _select_best_mask(results, canvas_bbox: list[int], canvas_size: int) -> np.ndarray | None:
    """
    Selecciona la mejor máscara con score compuesto.
    Las coordenadas ya están en espacio del canvas.

    Score = coverage × overlap^0.5 × centroid_bonus
    """
    x1, y1, x2, y2 = canvas_bbox

    masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
    mH, mW = masks_data.shape[1], masks_data.shape[2]

    # Escalar bbox a dimensiones de las máscaras internas
    sx, sy = mW / canvas_size, mH / canvas_size
    mx1, my1 = int(x1 * sx), int(y1 * sy)
    mx2, my2 = int(x2 * sx), int(y2 * sy)
    bbox_mask = np.zeros((mH, mW), dtype=bool)
    bbox_mask[my1:my2, mx1:mx2] = True
    bbox_area = bbox_mask.sum()

    cx = (mx1 + mx2) // 2
    cy = (my1 + my2) // 2

    best_score = -1.0
    best_idx = 0

    for i in range(masks_data.shape[0]):
        m = masks_data[i] > 0.5
        mask_area = m.sum()
        if mask_area == 0:
            continue

        inter = np.logical_and(m, bbox_mask).sum()
        coverage = inter / (bbox_area + 1e-6)
        overlap = (inter / (mask_area + 1e-6)) ** 0.5
        centroid_bonus = 1.2 if m[cy, cx] else 0.8
        score = coverage * overlap * centroid_bonus

        if score > best_score:
            best_score = score
            best_idx = i

    best = (masks_data[best_idx] > 0.5).astype(np.uint8) * 255
    print(f"✅ Máscara manual seleccionada (idx={best_idx}, score={best_score:.3f})")

    # Escalar al tamaño del canvas (INTER_NEAREST para binario)
    if best.shape[0] != canvas_size or best.shape[1] != canvas_size:
        best = cv2.resize(best, (canvas_size, canvas_size),
                          interpolation=cv2.INTER_NEAREST)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 4. POSTPROCESADO: revertir padding y escala al tamaño original
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_mask(mask_canvas: np.ndarray, meta: dict) -> np.ndarray:
    """
    Revierte el preprocesado: elimina padding y escala la máscara
    al tamaño original de la imagen.

    Usa INTER_NEAREST para preservar bordes binarios sin artefactos.
    """
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    new_w = meta["new_w"]
    new_h = meta["new_h"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    # Recortar la región útil (sin padding)
    mask_cropped = mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    # Escalar al tamaño original
    mask_original = cv2.resize(mask_cropped, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    return mask_original


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEBUG: inspeccionar todas las máscaras candidatas
# ─────────────────────────────────────────────────────────────────────────────

def _debug_masks(results, canvas_bbox: list[int], canvas_size: int):
    """Imprime estadísticas de todas las máscaras para diagnóstico."""
    masks_data = results[0].masks.data.cpu().numpy()
    n_masks = masks_data.shape[0]
    x1, y1, x2, y2 = canvas_bbox
    mH, mW = masks_data.shape[1], masks_data.shape[2]
    sx, sy = mW / canvas_size, mH / canvas_size

    print(f"🔍 DEBUG: {n_masks} máscaras generadas, mask_shape=({mH},{mW})")
    for i in range(n_masks):
        m = masks_data[i] > 0.5
        area = m.sum()
        total = mH * mW

        # Cobertura dentro del bbox
        mx1, my1 = int(x1 * sx), int(y1 * sy)
        mx2, my2 = int(x2 * sx), int(y2 * sy)
        bbox_region = m[my1:my2, mx1:mx2]
        bbox_coverage = bbox_region.sum() / (bbox_region.size + 1e-6) if bbox_region.size > 0 else 0

        print(f"  mask[{i}]: area={area} ({100*area/total:.1f}%), "
              f"bbox_coverage={100*bbox_coverage:.1f}%")

    # Scores de confianza si están disponibles
    if results[0].boxes is not None and results[0].boxes.conf is not None:
        confs = results[0].boxes.conf.cpu().numpy()
        for i, c in enumerate(confs):
            print(f"  mask[{i}] conf={c:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL: preprocesado → inferencia → postprocesado → RGBA
# ─────────────────────────────────────────────────────────────────────────────

def segment_with_fastsam(
    pil_image: Image.Image,
    bbox: list[int],
    debug: bool = False,
) -> bytes:
    """
    Pipeline completo de segmentación con preprocesado controlado.

    1. Preprocesa: resize + padding centrado en canvas cuadrado.
    2. Ejecuta FastSAM sobre el canvas (no sobre la imagen raw).
    3. Selecciona máscara con point prompt → box prompt → score manual.
    4. Postprocesa: revierte padding/escala al tamaño original.
    5. Compone RGBA con fondo transparente.

    Args:
        pil_image: Imagen PIL (cualquier modo).
        bbox: [x1, y1, x2, y2] en píxeles absolutos de la imagen original.
        debug: Si True, imprime estadísticas de todas las máscaras.
    Returns:
        bytes de un PNG con fondo transparente.
    """
    w, h = pil_image.size
    rgb = pil_image.convert("RGB")
    img_np = np.array(rgb)

    print(f"📡 Pipeline FastSAM — bbox={bbox} sobre {w}×{h}")

    # ── 1. Preprocesado ──────────────────────────────────────────────────
    canvas, meta = preprocess_image(img_np, target_size=1024)
    print(f"   Canvas: {meta['canvas_size']}×{meta['canvas_size']}, "
          f"scale={meta['scale']:.3f}, pad=({meta['pad_x']},{meta['pad_y']})")

    # ── 2. Inferencia + selección de máscara ─────────────────────────────
    mask_canvas = run_fastsam(canvas, meta, bbox, debug=debug)

    # Fallback: si no se obtuvo máscara, recorte rectangular
    if mask_canvas is None:
        print("⚠️  Sin máscara válida — devolviendo recorte rectangular")
        mask_original = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask_original[y1:y2, x1:x2] = 255
    else:
        # ── 3. Postprocesado ─────────────────────────────────────────────
        mask_original = postprocess_mask(mask_canvas, meta)

    # Binarizar por seguridad (INTER_NEAREST debería mantener, pero aseguramos)
    mask_binary = (mask_original > 127).astype(np.uint8) * 255

    # ── 4. Componer RGBA ─────────────────────────────────────────────────
    mask_pil = Image.fromarray(mask_binary, mode="L")
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    result.paste(rgb, mask=mask_pil)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print(f"✅ Segmentación completada — {w}×{h}")
    return buf.getvalue()
