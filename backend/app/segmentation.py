import io
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from ultralytics import FastSAM


_DEBUG_SEG_DIR = Path(__file__).parent / "debug_segmentation"


def _save_seg_debug(prefix: str, images: dict[str, np.ndarray]) -> str:
    """
    Guarda imágenes intermedias de la segmentación en debug_segmentation/.
    Retorna el timestamp usado como prefijo para correlacionar archivos.
    """
    try:
        _DEBUG_SEG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        for name, img in images.items():
            if img is None:
                continue
            path = _DEBUG_SEG_DIR / f"{ts}_{prefix}_{name}.png"
            arr = img
            if arr.ndim == 2:
                cv2.imwrite(str(path), arr)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            elif arr.ndim == 3 and arr.shape[2] == 4:
                bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(str(path), bgra)
            else:
                cv2.imwrite(str(path), arr)
        print(f"💾 Debug seg: {ts}_{prefix}_*.png ({len(images)} archivos)")
        return ts
    except Exception as exc:
        print(f"⚠️  No se pudo guardar debug seg: {exc}")
        return ""

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
# 2. FILTRADO DE BORDES: descartar máscaras individuales que tocan el borde
# ─────────────────────────────────────────────────────────────────────────────

def _filter_border_masks(results, meta: dict, canvas_size: int):
    """
    Descarta máscaras individuales que tocan los bordes del contenido
    de la imagen original (no el padding del canvas).
    Si todas las máscaras tocan bordes, no se filtra ninguna para no
    perder el resultado.
    """
    import torch

    masks_data = results[0].masks.data  # (N, mH, mW)
    masks_np = masks_data.cpu().numpy()
    n_masks, mH, mW = masks_np.shape

    # Mapear bordes del contenido de imagen al espacio de las máscaras
    sx, sy = mW / canvas_size, mH / canvas_size
    left = int(meta["pad_x"] * sx)
    right = int((meta["pad_x"] + meta["new_w"]) * sx) - 1
    top = int(meta["pad_y"] * sy)
    bottom = int((meta["pad_y"] + meta["new_h"]) * sy) - 1

    # Clamp por seguridad
    left = max(0, min(left, mW - 1))
    right = max(0, min(right, mW - 1))
    top = max(0, min(top, mH - 1))
    bottom = max(0, min(bottom, mH - 1))

    keep = []
    for i in range(n_masks):
        m = masks_np[i] > 0.5
        touches = (
            m[top, left:right + 1].any() or      # borde superior
            m[bottom, left:right + 1].any() or    # borde inferior
            m[top:bottom + 1, left].any() or      # borde izquierdo
            m[top:bottom + 1, right].any()         # borde derecho
        )
        if touches:
            print(f"   🚫 Máscara {i} descartada (toca borde de imagen)")
        else:
            keep.append(i)

    if len(keep) == n_masks:
        return  # Ninguna tocaba bordes, no hay nada que filtrar

    if len(keep) == 0:
        print("⚠️  Todas las máscaras tocan bordes — se conservan todas")
        return  # No filtrar para no quedarnos sin resultado

    print(f"   Filtrado de bordes: {n_masks} → {len(keep)} máscaras")
    keep_t = torch.tensor(keep, dtype=torch.long)
    results[0].masks.data = masks_data[keep_t]
    if results[0].boxes is not None:
        results[0].boxes = results[0].boxes[keep_t]


# ─────────────────────────────────────────────────────────────────────────────
# 2b. FILTRADO POR CONTENCIÓN EN BBOX: descartar máscaras mayormente fuera
# ─────────────────────────────────────────────────────────────────────────────

def _filter_bbox_containment(results, canvas_bbox: list[int], canvas_size: int, min_inside_ratio: float = 0.55):
    """
    Descarta máscaras donde menos del min_inside_ratio de su área total
    cae dentro del bbox del usuario. Esto elimina objetos de fondo que
    solapan parcialmente el bbox pero no son el objeto objetivo.

    Una máscara del objeto real que sangre ligeramente fuera del bbox
    (imprecisión de FastSAM) típicamente tiene >70% dentro y pasa el filtro.

    Si todas las máscaras serían descartadas, no se filtra (protección).
    """
    import torch

    masks_data = results[0].masks.data  # (N, mH, mW)
    masks_np = masks_data.cpu().numpy()
    n_masks, mH, mW = masks_np.shape

    x1, y1, x2, y2 = canvas_bbox
    sx, sy = mW / canvas_size, mH / canvas_size
    mx1, my1 = int(x1 * sx), int(y1 * sy)
    mx2, my2 = int(x2 * sx), int(y2 * sy)

    # Región del bbox en espacio de máscaras
    bbox_mask = np.zeros((mH, mW), dtype=bool)
    bbox_mask[my1:my2, mx1:mx2] = True

    keep = []
    for i in range(n_masks):
        m = masks_np[i] > 0.5
        mask_area = m.sum()
        if mask_area == 0:
            continue
        inside_area = np.logical_and(m, bbox_mask).sum()
        inside_ratio = inside_area / mask_area
        if inside_ratio >= min_inside_ratio:
            keep.append(i)
        else:
            print(f"   🚫 Máscara {i} descartada (solo {100*inside_ratio:.1f}% dentro del bbox)")

    if len(keep) == n_masks:
        return  # Todas pasan, no hay nada que filtrar

    if len(keep) == 0:
        print("⚠️  Todas las máscaras fuera del bbox — se conservan todas")
        return

    print(f"   Filtrado bbox containment: {n_masks} → {len(keep)} máscaras")
    keep_t = torch.tensor(keep, dtype=torch.long)
    results[0].masks.data = masks_data[keep_t]
    if results[0].boxes is not None:
        results[0].boxes = results[0].boxes[keep_t]


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFERENCIA: correr FastSAM sobre el canvas preprocesado
# ─────────────────────────────────────────────────────────────────────────────

def run_fastsam(
    canvas: np.ndarray,
    meta: dict,
    bbox: list[int],
    conf: float = 0.25,
    iou: float = 0.9,
    debug: bool = False,
    debug_imgs: dict | None = None,
    timings: dict | None = None,
) -> np.ndarray | None:
    """
    Ejecuta FastSAM sobre el canvas preprocesado y selecciona la mejor máscara.

    Estrategia de prompts:
      1. Box prompt (IoU) — robusto para formas irregulares.
      2. Fallback a point prompt (centro del bbox).
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
    _t_infer = time.perf_counter()
    results = model(
        canvas,
        device="cpu",
        retina_masks=True,
        imgsz=canvas_size,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    if timings is not None:
        timings["inference_ms"] = (time.perf_counter() - _t_infer) * 1000.0

    if not results or len(results) == 0 or results[0].masks is None:
        print("⚠️  FastSAM no generó máscaras")
        return None

    # ── Guardar todas las máscaras crudas devueltas por FastSAM ──────────
    if debug_imgs is not None:
        try:
            raw_masks = results[0].masks.data.cpu().numpy()
            print(f"💾 Guardando {raw_masks.shape[0]} máscaras crudas de FastSAM "
                  f"(shape={raw_masks.shape})")
            for i in range(raw_masks.shape[0]):
                m = (raw_masks[i] > 0.5).astype(np.uint8) * 255
                if m.shape[0] != canvas_size or m.shape[1] != canvas_size:
                    m = cv2.resize(m, (canvas_size, canvas_size),
                                   interpolation=cv2.INTER_NEAREST)
                # superponer bbox para referencia
                m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(m_bgr, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 0), 2)
                debug_imgs[f"02_raw_mask_{i:02d}"] = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            print(f"⚠️  No se pudieron guardar máscaras crudas: {exc}")

    # ── Filtrar máscaras individuales que tocan bordes de la imagen ──────
    _filter_border_masks(results, meta, canvas_size)
    if results[0].masks is None or results[0].masks.data.shape[0] == 0:
        print("⚠️  Todas las máscaras tocaban bordes — ninguna disponible")
        return None

    # ── Filtrar máscaras mayormente fuera del bbox del usuario ────────────
    _filter_bbox_containment(results, cb, canvas_size)
    if results[0].masks is None or results[0].masks.data.shape[0] == 0:
        print("⚠️  Todas las máscaras fuera del bbox — ninguna disponible")
        return None

    if debug:
        _debug_masks(results, cb, canvas_size)

    # ── Intento 1 (preferente): Unión de todas las máscaras contenidas en
    #    el bbox.  FastSAM suele devolver el objeto fragmentado en partes
    #    (cabeza/torso/piernas/base separadas); seleccionar una sola con
    #    box_prompt deja partes fuera.  La unión de las piezas que están
    #    mayoritariamente dentro del bbox reconstruye el objeto completo.
    mask = _try_union_strategy(results, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 2: Box prompt nativo de FastSAM ──────────────────────────
    mask = _try_native_box_prompt(canvas, results, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 3: Box prompt manual (IoU) ───────────────────────────────
    mask = _try_box_prompt(canvas, results, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 4: Point prompt (fallback para objetos centrados) ────────
    mask = _try_point_prompt(canvas, results, pt_x, pt_y, cb, canvas_size)
    if mask is not None:
        return mask

    # ── Intento 5: Selección manual por score compuesto ──────────────────
    print("⚠️  Prompts fallaron, selección manual por score")
    return _select_best_mask(results, cb, canvas_size, canvas)


def _try_union_strategy(
    results,
    canvas_bbox: list[int],
    canvas_size: int,
    min_inside_ratio: float = 0.60,
    max_canvas_coverage: float = 0.55,
    min_bbox_coverage: float = 0.20,
) -> np.ndarray | None:
    """
    Une todas las máscaras que están mayoritariamente dentro del bbox para
    reconstruir un objeto que FastSAM ha devuelto fragmentado.

    Filtros aplicados a cada máscara antes de unirla:
      - Área no nula.
      - Área < max_canvas_coverage del canvas total → descartar máscaras
        de "escena/fondo" que abarcan la mayor parte de la imagen.
      - >= min_inside_ratio del área de la máscara cae dentro del bbox →
        descartar objetos que solapen marginalmente el bbox pero no son
        parte del objeto objetivo.

    El resultado se valida con _mask_covers_bbox para evitar devolver
    máscaras que apenas cubren una esquina del bbox.
    """
    masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
    n_masks, mH, mW = masks_data.shape
    canvas_total = mH * mW

    x1, y1, x2, y2 = canvas_bbox
    sx, sy = mW / canvas_size, mH / canvas_size
    mx1, my1 = int(x1 * sx), int(y1 * sy)
    mx2, my2 = int(x2 * sx), int(y2 * sy)
    bbox_mask = np.zeros((mH, mW), dtype=bool)
    bbox_mask[my1:my2, mx1:mx2] = True

    union = np.zeros((mH, mW), dtype=bool)
    used = []
    for i in range(n_masks):
        m = masks_data[i] > 0.5
        mask_area = int(m.sum())
        if mask_area == 0:
            continue
        if mask_area / canvas_total > max_canvas_coverage:
            print(f"   ↪ union: mask[{i}] descartada — cubre "
                  f"{100*mask_area/canvas_total:.1f}% del canvas (escena/fondo)")
            continue
        inside = int(np.logical_and(m, bbox_mask).sum())
        inside_ratio = inside / mask_area
        if inside_ratio < min_inside_ratio:
            continue
        union |= m
        used.append((i, inside_ratio, mask_area))

    if not used:
        print("   ↪ union: ninguna máscara apta")
        return None

    print(f"   ✅ Union de {len(used)} máscara(s):")
    for idx, ratio, area in used:
        print(f"      mask[{idx}]: {area}px ({100*ratio:.1f}% dentro del bbox)")

    mask_bin = (union.astype(np.uint8) * 255)
    if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
        mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                              interpolation=cv2.INTER_NEAREST)

    if not _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=min_bbox_coverage):
        print("   ⚠️  union: cobertura del bbox insuficiente — fallback a otros prompts")
        return None

    print("✅ Máscara obtenida vía union_strategy")
    return mask_bin


def _try_point_prompt(canvas, results, pt_x, pt_y, canvas_bbox, canvas_size):
    """Selecciona la máscara que contiene el punto dado (centro del bbox)."""
    try:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
        n_masks, mH, mW = masks_data.shape

        # Mapear punto al espacio de las máscaras internas
        sx, sy = mW / canvas_size, mH / canvas_size
        px, py = int(pt_x * sx), int(pt_y * sy)
        px = max(0, min(px, mW - 1))
        py = max(0, min(py, mH - 1))

        # Buscar la mejor máscara que contenga el punto
        best_idx = -1
        best_score = -1.0
        for i in range(n_masks):
            m = masks_data[i] > 0.5
            if not m[py, px]:
                continue
            # Preferir máscaras más pequeñas (más específicas)
            area = m.sum()
            score = 1.0 / (area + 1e-6)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            print("⚠️  point_prompt: ninguna máscara contiene el punto")
            return None

        mask_bin = (masks_data[best_idx] > 0.5).astype(np.uint8) * 255
        if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
            mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                                  interpolation=cv2.INTER_NEAREST)

        if _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=0.20):
            print("✅ Máscara obtenida vía point_prompt")
            return mask_bin
        else:
            print("⚠️  point_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"⚠️  point_prompt falló: {e}")
    return None


def _try_native_box_prompt(canvas, results, canvas_bbox, canvas_size):
    """Usa FastSAMPrompt.box_prompt (API nativa de ultralytics) para seleccionar la máscara."""
    try:
        from ultralytics.models.fastsam import FastSAMPrompt

        prompt_process = FastSAMPrompt(canvas, results, device="cpu")
        ann = prompt_process.box_prompt(bboxes=[canvas_bbox])

        if ann is None or (hasattr(ann, "__len__") and len(ann) == 0):
            print("⚠️  native_box_prompt: sin resultado")
            return None

        ann = np.array(ann)
        mask_bin = ann[0] if ann.ndim == 3 else ann
        mask_bin = (mask_bin > 0.5).astype(np.uint8) * 255

        if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
            mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                                  interpolation=cv2.INTER_NEAREST)

        if _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=0.15):
            print("✅ Máscara obtenida vía native_box_prompt")
            return mask_bin
        else:
            print("⚠️  native_box_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"⚠️  native_box_prompt falló: {e}")
    return None


def _try_box_prompt(canvas, results, canvas_bbox, canvas_size):
    """Selecciona la máscara con mayor IoU respecto al bbox."""
    try:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
        n_masks, mH, mW = masks_data.shape
        x1, y1, x2, y2 = canvas_bbox

        # Mapear bbox al espacio de las máscaras internas
        sx, sy = mW / canvas_size, mH / canvas_size
        mx1, my1 = int(x1 * sx), int(y1 * sy)
        mx2, my2 = int(x2 * sx), int(y2 * sy)
        bbox_mask = np.zeros((mH, mW), dtype=bool)
        bbox_mask[my1:my2, mx1:mx2] = True
        bbox_area = bbox_mask.sum()

        best_idx = -1
        best_iou = -1.0
        for i in range(n_masks):
            m = masks_data[i] > 0.5
            inter = np.logical_and(m, bbox_mask).sum()
            union = np.logical_or(m, bbox_mask).sum()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx < 0:
            return None

        mask_bin = (masks_data[best_idx] > 0.5).astype(np.uint8) * 255
        if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
            mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                                  interpolation=cv2.INTER_NEAREST)

        if _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=0.15):
            print(f"✅ Máscara obtenida vía box_prompt (IoU={best_iou:.3f})")
            return mask_bin
        else:
            print("⚠️  box_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"⚠️  box_prompt falló: {e}")
    return None



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

def _select_best_mask(results, canvas_bbox: list[int], canvas_size: int, canvas: np.ndarray = None) -> np.ndarray | None:
    """
    Selecciona la mejor máscara con score compuesto.
    Las coordenadas ya están en espacio del canvas.

    Score = coverage × overlap^0.5 × centroid_bonus × edge_bonus
    El edge_bonus usa Canny para premiar máscaras cuyos bordes coinciden
    con bordes reales de la imagen (30% de peso).
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

        # Edge-guided bonus: premiar máscaras alineadas con bordes reales
        if canvas is not None:
            mask_resized = (m.astype(np.uint8) * 255)
            if mask_resized.shape[0] != canvas_size or mask_resized.shape[1] != canvas_size:
                mask_resized = cv2.resize(mask_resized, (canvas_size, canvas_size),
                                          interpolation=cv2.INTER_NEAREST)
            edge_score = _edge_guided_score(canvas, mask_resized, canvas_bbox, canvas_size)
            score = score * (0.7 + 0.3 * edge_score)

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


def _antialias_mask(mask_binary: np.ndarray, scale: float) -> np.ndarray:
    """
    Suaviza los bordes "en escalones" de la máscara binaria convirtiéndola en
    un alpha con antialiasing (gradiente 0..255 en el borde).

    La máscara se calcula a ~1024px y se reescala al tamaño original con
    INTER_NEAREST, lo que produce un borde dentado cuyos "escalones" miden
    aproximadamente 1/scale píxeles. Aplicamos un desenfoque gaussiano
    proporcional a ese tamaño de escalón para fundir el dentado sin
    redondear en exceso la silueta.

    Args:
        mask_binary: máscara HxW uint8 (0 o 255).
        scale: meta["scale"] = tamaño_canvas / lado_mayor_original (<= 1).
    Returns:
        Máscara HxW uint8 con bordes suavizados (alpha antialiased).
    """
    step_px = 1.0 / max(scale, 1e-3)          # px originales por píxel de máscara
    sigma = max(0.8, 0.6 * step_px)
    soft = cv2.GaussianBlur(mask_binary, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return soft


def _clip_mask_to_bbox(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    """
    Recorta la máscara estrictamente al bbox del usuario.
    Todos los píxeles fuera del bbox se ponen a 0.

    Esto garantiza que el resultado final de segmentación esté siempre
    contenido dentro del bounding box manual.

    Args:
        mask: Máscara binaria (H, W), valores 0 o 255, coordenadas originales.
        bbox: [x1, y1, x2, y2] en píxeles de la imagen original.
    Returns:
        Máscara clipeada con misma forma.
    """
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]

    clipped = np.zeros_like(mask)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    clipped[y1c:y2c, x1c:x2c] = mask[y1c:y2c, x1c:x2c]
    return clipped


def _edge_guided_score(canvas: np.ndarray, mask_bin: np.ndarray, canvas_bbox: list[int], canvas_size: int) -> float:
    """
    Puntúa cuán bien los bordes de una máscara se alinean con bordes reales
    detectados por Canny dentro del bbox.

    Pasos:
    1. Extraer región del bbox del canvas RGB.
    2. Canny edge detection con umbrales adaptativos.
    3. Dilatar bordes para zona de tolerancia.
    4. Calcular contorno de la máscara candidata en la región del bbox.
    5. Medir qué % del contorno coincide con bordes Canny.

    Returns:
        Score entre 0.0 y 1.0. Mayor = mejor alineamiento con bordes reales.
    """
    x1, y1, x2, y2 = canvas_bbox
    # Clamp por seguridad
    h_canvas, w_canvas = canvas.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_canvas, x2), min(h_canvas, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = canvas[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    median_val = np.median(blurred)
    low = int(max(0, 0.66 * median_val))
    high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, low, high)

    # Dilatar bordes para tolerancia de 2px
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_zone = cv2.dilate(edges, kernel, iterations=1)

    # Obtener contorno de la máscara dentro de la región del bbox
    # La máscara está en espacio de canvas completo, extraer la región del bbox
    mask_roi = mask_bin[y1:y2, x1:x2]
    mask_u8 = (mask_roi > 127).astype(np.uint8) if mask_roi.max() > 1 else mask_roi.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return 0.0

    # Dibujar contorno como línea de 1px
    contour_img = np.zeros_like(mask_u8)
    cv2.drawContours(contour_img, contours, -1, 1, 1)

    contour_pixels = contour_img.sum()
    if contour_pixels == 0:
        return 0.0

    # Cuántos píxeles del contorno caen sobre bordes detectados
    aligned = np.logical_and(contour_img > 0, edge_zone > 0).sum()
    return float(aligned / contour_pixels)


def _remove_border_components(mask: np.ndarray) -> np.ndarray:
    """
    Elimina componentes conexos de la máscara binaria que tocan cualquier
    borde de la imagen. Esto descarta fragmentos espurios (pelo, fondo, etc.)
    que se extienden hasta el borde, sin descartar el objeto principal que
    normalmente está centrado.

    Si TODOS los componentes tocan bordes, devuelve la máscara sin cambios
    para no perder el resultado.
    """
    h, w = mask.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(mask_bin)
    # label 0 = fondo, componentes reales van de 1 a n_labels-1
    if n_labels <= 1:
        return mask  # Sin componentes, nada que filtrar

    # Crear máscara de borde: fila 0, fila H-1, col 0, col W-1
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[h - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, w - 1] = True

    # Identificar qué labels tocan el borde
    border_labels = set(np.unique(labels[border_mask]))
    border_labels.discard(0)  # Ignorar fondo

    if len(border_labels) == 0:
        return mask  # Ningún componente toca bordes

    interior_labels = set(range(1, n_labels)) - border_labels

    if len(interior_labels) == 0:
        print("⚠️  Todos los componentes tocan bordes — se conserva máscara completa")
        return mask

    # Reconstruir máscara solo con componentes interiores
    clean = np.zeros_like(mask_bin)
    for lbl in interior_labels:
        clean[labels == lbl] = 1

    removed_px = mask_bin.sum() - clean.sum()
    print(f"   🧹 Bordes: {len(border_labels)} componente(s) eliminado(s), "
          f"{removed_px} píxeles removidos")

    return clean * 255


def _keep_largest_components(mask: np.ndarray, min_ratio: float = 0.10) -> np.ndarray:
    """
    Filtra fragmentos desconectados de la máscara, conservando solo el
    componente más grande y cualquier otro cuya área sea al menos
    `min_ratio` del componente principal.  Descarta fragmentos pequeños
    (ruido, artefactos) que no pertenecen al objeto.
    """
    mask_bin = (mask > 127).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(mask_bin)
    if n_labels <= 2:
        return mask  # 0 o 1 componente, nada que filtrar

    # Área de cada label (índice 0 = fondo)
    areas = np.bincount(labels.ravel())
    component_areas = areas[1:]  # solo componentes de primer plano

    max_area = component_areas.max()
    threshold = max_area * min_ratio

    # Labels a conservar (label = índice + 1 porque label 0 es fondo)
    keep_labels = {i + 1 for i, area in enumerate(component_areas)
                   if area >= threshold}

    if len(keep_labels) == n_labels - 1:
        return mask  # Todos los componentes son suficientemente grandes

    # Reconstruir máscara limpia
    clean = np.zeros_like(mask_bin)
    for lbl in keep_labels:
        clean[labels == lbl] = 1

    removed_count = (n_labels - 1) - len(keep_labels)
    removed_px = mask_bin.sum() - clean.sum()
    print(f"   🧩 Fragmentos: {removed_count} componente(s) pequeño(s) eliminado(s), "
          f"{removed_px} píxeles descartados (umbral={threshold:.0f}px, "
          f"{min_ratio * 100:.0f}% del principal)")

    return clean * 255


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
    target_size: int = 1024,
    timings: dict | None = None,
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
        target_size: Lado del canvas cuadrado de trabajo (letterbox). El canvas
            real se satura en min(target_size, lado_mayor_original), por lo que
            subir este valor por encima del lado mayor de la imagen no cambia la
            inferencia. Útil para medir latencia a distintas resoluciones.
        timings: Si se pasa un dict, se rellenan las claves preprocess_ms,
            inference_ms, postprocess_ms y total_ms con la latencia de cada fase.
    Returns:
        bytes de un PNG con fondo transparente.
    """
    w, h = pil_image.size
    rgb = pil_image.convert("RGB")
    img_np = np.array(rgb)

    print(f"📡 Pipeline FastSAM — bbox={bbox} sobre {w}×{h}")

    _local_timings: dict = {} if timings is None else timings
    _t_total = time.perf_counter()

    # ── 1. Preprocesado ──────────────────────────────────────────────────
    _t_pre = time.perf_counter()
    canvas, meta = preprocess_image(img_np, target_size=target_size)
    _local_timings["preprocess_ms"] = (time.perf_counter() - _t_pre) * 1000.0
    print(f"   Canvas: {meta['canvas_size']}×{meta['canvas_size']}, "
          f"scale={meta['scale']:.3f}, pad=({meta['pad_x']},{meta['pad_y']})")

    # Dibujar bbox sobre original y sobre canvas para debug
    orig_with_bbox = img_np.copy()
    cv2.rectangle(orig_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), max(2, w // 500))
    cb = _bbox_to_canvas(bbox, meta)
    canvas_with_bbox = canvas.copy()
    cv2.rectangle(canvas_with_bbox, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 0), 2)

    debug_imgs = {
        "00_original_bbox": orig_with_bbox,
        "01_canvas_bbox": canvas_with_bbox,
    }

    # ── 2. Inferencia + selección de máscara ─────────────────────────────
    # run_fastsam mide la inferencia pura del modelo en timings["inference_ms"];
    # el resto del tiempo de esta llamada (selección de máscara) se contabiliza
    # como postprocesado.
    _t_run = time.perf_counter()
    mask_canvas = run_fastsam(canvas, meta, bbox, debug=debug,
                              debug_imgs=debug_imgs, timings=_local_timings)
    _run_total_ms = (time.perf_counter() - _t_run) * 1000.0
    _selection_ms = _run_total_ms - _local_timings.get("inference_ms", 0.0)
    _t_post = time.perf_counter()

    # Fallback: si no se obtuvo máscara, recorte rectangular
    if mask_canvas is None:
        print("⚠️  Sin máscara válida — devolviendo recorte rectangular")
        mask_original = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask_original[y1:y2, x1:x2] = 255
    else:
        debug_imgs["10_mask_canvas_selected"] = mask_canvas
        # ── 3. Postprocesado ─────────────────────────────────────────────
        mask_original = postprocess_mask(mask_canvas, meta)
        debug_imgs["11_mask_original_postproc"] = mask_original

    # Binarizar por seguridad (INTER_NEAREST debería mantener, pero aseguramos)
    mask_binary = (mask_original > 127).astype(np.uint8) * 255

    # ── 3a. Clip estricto al bbox del usuario ────────────────────────────
    mask_binary = _clip_mask_to_bbox(mask_binary, bbox)
    debug_imgs["12_mask_after_clip_bbox"] = mask_binary

    # ── 3b. Eliminar componentes conexos que tocan bordes ────────────────
    mask_binary = _remove_border_components(mask_binary)
    debug_imgs["13_mask_after_remove_border"] = mask_binary

    # ── 3c. Filtrar fragmentos: conservar solo componentes grandes ───────
    mask_binary = _keep_largest_components(mask_binary)
    debug_imgs["14_mask_final"] = mask_binary

    # ── 3d. Antialiasing de bordes: suavizar el dentado del borde binario ─
    mask_alpha = _antialias_mask(mask_binary, meta["scale"])
    debug_imgs["14b_mask_antialiased"] = mask_alpha

    # ── 4. Componer RGBA ─────────────────────────────────────────────────
    mask_pil = Image.fromarray(mask_alpha, mode="L")
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    result.paste(rgb, mask=mask_pil)

    # Guardar overlay del resultado sobre fondo gris para inspección visual
    overlay = np.full((h, w, 3), 64, dtype=np.uint8)
    overlay[mask_binary > 0] = img_np[mask_binary > 0]
    debug_imgs["15_result_overlay"] = overlay

    _save_seg_debug("seg", debug_imgs)

    buf = io.BytesIO()
    result.save(buf, format="PNG")

    # postprocesado = selección de máscara (parte de run_fastsam) + revertir
    # padding + limpieza + composición RGBA.
    _local_timings["postprocess_ms"] = _selection_ms + (time.perf_counter() - _t_post) * 1000.0
    _local_timings["total_ms"] = (time.perf_counter() - _t_total) * 1000.0
    print(
        f"⏱️  Latencia — pre={_local_timings['preprocess_ms']:.1f} ms | "
        f"infer={_local_timings.get('inference_ms', 0.0):.1f} ms | "
        f"post={_local_timings['postprocess_ms']:.1f} ms | "
        f"total={_local_timings['total_ms']:.1f} ms "
        f"(canvas={meta['canvas_size']}px)"
    )

    print(f"✅ Segmentación completada — {w}×{h}")
    return buf.getvalue()
