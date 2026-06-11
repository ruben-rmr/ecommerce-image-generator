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
    Guarda las imágenes intermedias de la segmentación en debug_segmentation/.
    Devuelve el timestamp usado como prefijo, que sirve para correlacionar los archivos.
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
        print(f"Debug seg: {ts}_{prefix}_*.png ({len(images)} archivos)")
        return ts
    except Exception as exc:
        print(f"No se pudo guardar debug seg: {exc}")
        return ""


# --- Carga del modelo FastSAM local (una sola vez al importar el módulo) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "FastSAM-s.pt")
_model = None


def _get_model():
    global _model
    if _model is None:
        print(f"Cargando FastSAM desde {MODEL_PATH} ...")
        _model = FastSAM(MODEL_PATH)
        print("Modelo FastSAM cargado.")
    return _model


# --- 1. Preprocesado: resize controlado + padding centrado en lienzo cuadrado ---

def preprocess_image(img_np: np.ndarray, target_size: int = 1024) -> tuple[np.ndarray, dict]:
    """
    Escala la imagen manteniendo la relación de aspecto y la centra en un lienzo cuadrado
    con relleno gris (114, 114, 114), el mismo color que usa el letterbox de YOLO.

    Devuelve el lienzo (imagen cuadrada target_size x target_size x 3, uint8 RGB) y un dict
    `meta` con scale, pad_x, pad_y, orig_h y orig_w para poder revertir la transformación.
    """
    h, w = img_np.shape[:2]

    # Escalamos por el lado mayor, evitando un upscaling agresivo.
    effective_size = min(target_size, max(h, w))
    scale = effective_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Lienzo cuadrado con el contenido centrado.
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
    """Pasa un bbox [x1,y1,x2,y2] de coordenadas originales a coordenadas del lienzo."""
    scale, pad_x, pad_y = meta["scale"], meta["pad_x"], meta["pad_y"]
    x1 = int(bbox[0] * scale) + pad_x
    y1 = int(bbox[1] * scale) + pad_y
    x2 = int(bbox[2] * scale) + pad_x
    y2 = int(bbox[3] * scale) + pad_y
    return [x1, y1, x2, y2]


def _point_to_canvas(x: int, y: int, meta: dict) -> tuple[int, int]:
    """Pasa un punto de coordenadas originales a coordenadas del lienzo."""
    cx = int(x * meta["scale"]) + meta["pad_x"]
    cy = int(y * meta["scale"]) + meta["pad_y"]
    return cx, cy


# --- 2. Filtrado de bordes: descartar máscaras que tocan el borde del contenido ---

def _filter_border_masks(results, meta: dict, canvas_size: int):
    """
    Descarta las máscaras individuales que tocan los bordes del contenido de la imagen
    original (no el relleno del lienzo). Si todas las máscaras tocan algún borde, no se
    filtra ninguna para no quedarse sin resultado.
    """
    import torch

    masks_data = results[0].masks.data  # (N, mH, mW)
    masks_np = masks_data.cpu().numpy()
    n_masks, mH, mW = masks_np.shape

    # Mapeamos los bordes del contenido al espacio de las máscaras.
    sx, sy = mW / canvas_size, mH / canvas_size
    left = int(meta["pad_x"] * sx)
    right = int((meta["pad_x"] + meta["new_w"]) * sx) - 1
    top = int(meta["pad_y"] * sy)
    bottom = int((meta["pad_y"] + meta["new_h"]) * sy) - 1

    # Recorte de seguridad.
    left = max(0, min(left, mW - 1))
    right = max(0, min(right, mW - 1))
    top = max(0, min(top, mH - 1))
    bottom = max(0, min(bottom, mH - 1))

    keep = []
    for i in range(n_masks):
        m = masks_np[i] > 0.5
        touches = (
            m[top, left:right + 1].any() or       # borde superior
            m[bottom, left:right + 1].any() or    # borde inferior
            m[top:bottom + 1, left].any() or      # borde izquierdo
            m[top:bottom + 1, right].any()        # borde derecho
        )
        if touches:
            print(f"   Máscara {i} descartada (toca borde de imagen)")
        else:
            keep.append(i)

    if len(keep) == n_masks:
        return  # Ninguna tocaba bordes, no hay nada que filtrar.

    if len(keep) == 0:
        print("Todas las máscaras tocan bordes — se conservan todas")
        return  # No filtramos para no quedarnos sin resultado.

    print(f"   Filtrado de bordes: {n_masks} -> {len(keep)} máscaras")
    keep_t = torch.tensor(keep, dtype=torch.long)
    results[0].masks.data = masks_data[keep_t]
    if results[0].boxes is not None:
        results[0].boxes = results[0].boxes[keep_t]


# --- 2b. Filtrado por contención en el bbox: descartar máscaras mayormente fuera ---

def _filter_bbox_containment(results, canvas_bbox: list[int], canvas_size: int, min_inside_ratio: float = 0.55):
    """
    Descarta las máscaras en las que menos del min_inside_ratio de su área total cae dentro
    del bbox del usuario. Así se eliminan objetos de fondo que solapan parcialmente el bbox
    pero no son el objeto buscado.

    Una máscara del objeto real que se salga un poco del bbox (imprecisión de FastSAM) suele
    tener más del 70 % dentro y pasa el filtro. Si se descartarían todas, no se filtra nada.
    """
    import torch

    masks_data = results[0].masks.data  # (N, mH, mW)
    masks_np = masks_data.cpu().numpy()
    n_masks, mH, mW = masks_np.shape

    x1, y1, x2, y2 = canvas_bbox
    sx, sy = mW / canvas_size, mH / canvas_size
    mx1, my1 = int(x1 * sx), int(y1 * sy)
    mx2, my2 = int(x2 * sx), int(y2 * sy)

    # Región del bbox en el espacio de las máscaras.
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
            print(f"   Máscara {i} descartada (solo {100*inside_ratio:.1f}% dentro del bbox)")

    if len(keep) == n_masks:
        return  # Pasan todas, no hay nada que filtrar.

    if len(keep) == 0:
        print("Todas las máscaras fuera del bbox — se conservan todas")
        return

    print(f"   Filtrado por contención en bbox: {n_masks} -> {len(keep)} máscaras")
    keep_t = torch.tensor(keep, dtype=torch.long)
    results[0].masks.data = masks_data[keep_t]
    if results[0].boxes is not None:
        results[0].boxes = results[0].boxes[keep_t]


# --- 3. Inferencia: ejecutar FastSAM sobre el lienzo preprocesado ---

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
    Ejecuta FastSAM sobre el lienzo preprocesado y selecciona la mejor máscara.

    Las estrategias de prompt se prueban en cascada: unión de fragmentos contenidos en el
    bbox, box prompt nativo de FastSAM, box prompt manual por IoU, point prompt y, como
    último recurso, selección manual por score compuesto.

    Devuelve una máscara binaria en coordenadas del lienzo (canvas_size x canvas_size), o None.
    """
    model = _get_model()
    canvas_size = meta["canvas_size"]

    # Bbox ya transformado al espacio del lienzo.
    cb = _bbox_to_canvas(bbox, meta)

    # Centro del bbox original -> punto en el lienzo.
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    pt_x, pt_y = _point_to_canvas(center_x, center_y, meta)

    print(f"FastSAM inferencia: canvas={canvas_size}x{canvas_size}, "
          f"bbox_canvas={cb}, point=({pt_x},{pt_y})")

    # imgsz = canvas_size porque ya hemos hecho el preprocesado nosotros.
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
        print("FastSAM no generó máscaras")
        return None

    # Guardamos todas las máscaras crudas que devuelve FastSAM.
    if debug_imgs is not None:
        try:
            raw_masks = results[0].masks.data.cpu().numpy()
            print(f"Guardando {raw_masks.shape[0]} máscaras crudas de FastSAM "
                  f"(shape={raw_masks.shape})")
            for i in range(raw_masks.shape[0]):
                m = (raw_masks[i] > 0.5).astype(np.uint8) * 255
                if m.shape[0] != canvas_size or m.shape[1] != canvas_size:
                    m = cv2.resize(m, (canvas_size, canvas_size),
                                   interpolation=cv2.INTER_NEAREST)
                # Superponemos el bbox como referencia.
                m_bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(m_bgr, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 0), 2)
                debug_imgs[f"02_raw_mask_{i:02d}"] = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            print(f"No se pudieron guardar las máscaras crudas: {exc}")

    # Filtramos las máscaras individuales que tocan los bordes de la imagen.
    _filter_border_masks(results, meta, canvas_size)
    if results[0].masks is None or results[0].masks.data.shape[0] == 0:
        print("Todas las máscaras tocaban bordes — ninguna disponible")
        return None

    # Filtramos las máscaras que quedan mayormente fuera del bbox del usuario.
    _filter_bbox_containment(results, cb, canvas_size)
    if results[0].masks is None or results[0].masks.data.shape[0] == 0:
        print("Todas las máscaras fuera del bbox — ninguna disponible")
        return None

    if debug:
        _debug_masks(results, cb, canvas_size)

    # Intento 1 (preferente): unión de todas las máscaras contenidas en el bbox. FastSAM
    # suele devolver el objeto fragmentado en partes (cabeza/torso/piernas/base por
    # separado); quedarse con una sola vía box_prompt deja trozos fuera. La unión de las
    # piezas que están mayoritariamente dentro del bbox reconstruye el objeto completo.
    mask = _try_union_strategy(results, cb, canvas_size)
    if mask is not None:
        return mask

    # Intento 2: box prompt nativo de FastSAM.
    mask = _try_native_box_prompt(canvas, results, cb, canvas_size)
    if mask is not None:
        return mask

    # Intento 3: box prompt manual (IoU).
    mask = _try_box_prompt(canvas, results, cb, canvas_size)
    if mask is not None:
        return mask

    # Intento 4: point prompt (fallback para objetos centrados).
    mask = _try_point_prompt(canvas, results, pt_x, pt_y, cb, canvas_size)
    if mask is not None:
        return mask

    # Intento 5: selección manual por score compuesto.
    print("Los prompts fallaron; selección manual por score")
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
    Une todas las máscaras que están mayoritariamente dentro del bbox para reconstruir un
    objeto que FastSAM ha devuelto fragmentado.

    Antes de unir cada máscara se aplican varios filtros: que su área no sea nula, que ocupe
    menos de max_canvas_coverage del lienzo (descarta máscaras de "escena/fondo" que cubren
    casi toda la imagen) y que al menos min_inside_ratio de su área caiga dentro del bbox
    (descarta objetos que solo lo rozan). El resultado se valida con _mask_covers_bbox para
    no devolver máscaras que apenas cubren una esquina del bbox.
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
            print(f"   union: mask[{i}] descartada — cubre "
                  f"{100*mask_area/canvas_total:.1f}% del lienzo (escena/fondo)")
            continue
        inside = int(np.logical_and(m, bbox_mask).sum())
        inside_ratio = inside / mask_area
        if inside_ratio < min_inside_ratio:
            continue
        union |= m
        used.append((i, inside_ratio, mask_area))

    if not used:
        print("   union: ninguna máscara apta")
        return None

    print(f"   Unión de {len(used)} máscara(s):")
    for idx, ratio, area in used:
        print(f"      mask[{idx}]: {area}px ({100*ratio:.1f}% dentro del bbox)")

    mask_bin = (union.astype(np.uint8) * 255)
    if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
        mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                              interpolation=cv2.INTER_NEAREST)

    if not _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=min_bbox_coverage):
        print("   union: cobertura del bbox insuficiente — se prueban otros prompts")
        return None

    print("Máscara obtenida vía union_strategy")
    return mask_bin


def _try_point_prompt(canvas, results, pt_x, pt_y, canvas_bbox, canvas_size):
    """Selecciona la máscara que contiene el punto dado (el centro del bbox)."""
    try:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
        n_masks, mH, mW = masks_data.shape

        # Mapeamos el punto al espacio interno de las máscaras.
        sx, sy = mW / canvas_size, mH / canvas_size
        px, py = int(pt_x * sx), int(pt_y * sy)
        px = max(0, min(px, mW - 1))
        py = max(0, min(py, mH - 1))

        # Buscamos la mejor máscara que contenga el punto.
        best_idx = -1
        best_score = -1.0
        for i in range(n_masks):
            m = masks_data[i] > 0.5
            if not m[py, px]:
                continue
            # Preferimos máscaras más pequeñas (más específicas).
            area = m.sum()
            score = 1.0 / (area + 1e-6)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            print("point_prompt: ninguna máscara contiene el punto")
            return None

        mask_bin = (masks_data[best_idx] > 0.5).astype(np.uint8) * 255
        if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
            mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                                  interpolation=cv2.INTER_NEAREST)

        if _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=0.20):
            print("Máscara obtenida vía point_prompt")
            return mask_bin
        else:
            print("point_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"point_prompt falló: {e}")
    return None


def _try_native_box_prompt(canvas, results, canvas_bbox, canvas_size):
    """Usa FastSAMPrompt.box_prompt (API nativa de ultralytics) para elegir la máscara."""
    try:
        from ultralytics.models.fastsam import FastSAMPrompt

        prompt_process = FastSAMPrompt(canvas, results, device="cpu")
        ann = prompt_process.box_prompt(bboxes=[canvas_bbox])

        if ann is None or (hasattr(ann, "__len__") and len(ann) == 0):
            print("native_box_prompt: sin resultado")
            return None

        ann = np.array(ann)
        mask_bin = ann[0] if ann.ndim == 3 else ann
        mask_bin = (mask_bin > 0.5).astype(np.uint8) * 255

        if mask_bin.shape[0] != canvas_size or mask_bin.shape[1] != canvas_size:
            mask_bin = cv2.resize(mask_bin, (canvas_size, canvas_size),
                                  interpolation=cv2.INTER_NEAREST)

        if _mask_covers_bbox(mask_bin, canvas_bbox, min_coverage=0.15):
            print("Máscara obtenida vía native_box_prompt")
            return mask_bin
        else:
            print("native_box_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"native_box_prompt falló: {e}")
    return None


def _try_box_prompt(canvas, results, canvas_bbox, canvas_size):
    """Selecciona la máscara con mayor IoU respecto al bbox."""
    try:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
        n_masks, mH, mW = masks_data.shape
        x1, y1, x2, y2 = canvas_bbox

        # Mapeamos el bbox al espacio interno de las máscaras.
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
            print(f"Máscara obtenida vía box_prompt (IoU={best_iou:.3f})")
            return mask_bin
        else:
            print("box_prompt: máscara insuficiente (baja cobertura)")
    except Exception as e:
        print(f"box_prompt falló: {e}")
    return None


def _mask_covers_bbox(mask: np.ndarray, bbox: list[int], min_coverage: float) -> bool:
    """Comprueba que la máscara cubre al menos min_coverage del área del bbox."""
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bbox_region = mask[y1:y2, x1:x2]
    if bbox_region.size == 0:
        return False
    coverage = (bbox_region > 0).sum() / bbox_region.size
    return coverage >= min_coverage


# --- 3b. Selección manual (fallback): score compuesto coverage x overlap x centroid ---

def _select_best_mask(results, canvas_bbox: list[int], canvas_size: int, canvas: np.ndarray = None) -> np.ndarray | None:
    """
    Selecciona la mejor máscara mediante un score compuesto. Las coordenadas ya están en el
    espacio del lienzo.

    Score = coverage x overlap^0.5 x centroid_bonus x edge_bonus. El edge_bonus usa Canny
    para premiar las máscaras cuyos bordes coinciden con bordes reales de la imagen (con un
    peso del 30 %).
    """
    x1, y1, x2, y2 = canvas_bbox

    masks_data = results[0].masks.data.cpu().numpy()  # (N, mH, mW)
    mH, mW = masks_data.shape[1], masks_data.shape[2]

    # Escalamos el bbox a las dimensiones internas de las máscaras.
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

        # Bonus guiado por bordes: premia las máscaras alineadas con bordes reales.
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
    print(f"Máscara manual seleccionada (idx={best_idx}, score={best_score:.3f})")

    # Escalamos al tamaño del lienzo (INTER_NEAREST para mantenerla binaria).
    if best.shape[0] != canvas_size or best.shape[1] != canvas_size:
        best = cv2.resize(best, (canvas_size, canvas_size),
                          interpolation=cv2.INTER_NEAREST)
    return best


# --- 4. Postprocesado: revertir relleno y escala al tamaño original ---

def postprocess_mask(mask_canvas: np.ndarray, meta: dict) -> np.ndarray:
    """
    Revierte el preprocesado: quita el relleno y escala la máscara al tamaño original de la
    imagen. Usa INTER_NEAREST para conservar los bordes binarios sin artefactos.
    """
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    new_w = meta["new_w"]
    new_h = meta["new_h"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    # Recortamos la región útil (sin el relleno).
    mask_cropped = mask_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]

    # Escalamos al tamaño original.
    mask_original = cv2.resize(mask_cropped, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    return mask_original


def _antialias_mask(mask_binary: np.ndarray, scale: float) -> np.ndarray:
    """
    Suaviza el borde "en escalones" de la máscara binaria convirtiéndolo en un alfa con
    antialiasing (gradiente 0..255 en el borde).

    La máscara se calcula a ~1024px y se reescala al tamaño original con INTER_NEAREST, lo que
    deja un borde dentado cuyos escalones miden aproximadamente 1/scale píxeles. Aplicamos un
    desenfoque gaussiano proporcional a ese tamaño de escalón para fundir el dentado sin
    redondear en exceso la silueta.

    `scale` es meta["scale"] = tamaño_lienzo / lado_mayor_original (<= 1). Devuelve la máscara
    HxW uint8 con los bordes suavizados.
    """
    step_px = 1.0 / max(scale, 1e-3)          # px originales por píxel de máscara
    sigma = max(0.8, 0.6 * step_px)
    soft = cv2.GaussianBlur(mask_binary, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return soft


def _clip_mask_to_bbox(mask: np.ndarray, bbox: list[int]) -> np.ndarray:
    """
    Recorta la máscara estrictamente al bbox del usuario: todos los píxeles fuera del bbox se
    ponen a 0. Garantiza que el resultado final de la segmentación siempre quede contenido
    dentro del bounding box manual.

    `mask` es la máscara binaria (H, W) con valores 0 o 255 en coordenadas originales y `bbox`
    es [x1, y1, x2, y2] en píxeles de la imagen original.
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
    Puntúa cómo de bien encajan los bordes de una máscara con los bordes reales detectados por
    Canny dentro del bbox.

    El proceso es: extraer la región del bbox del lienzo RGB, detectar bordes con Canny de
    umbrales adaptativos, dilatarlos para crear una zona de tolerancia, calcular el contorno de
    la máscara candidata y medir qué porcentaje de ese contorno coincide con los bordes de Canny.

    Devuelve un score entre 0.0 y 1.0 (más alto = mejor alineamiento con los bordes reales).
    """
    x1, y1, x2, y2 = canvas_bbox
    # Recorte de seguridad.
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

    # Dilatamos los bordes para una tolerancia de 2px.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_zone = cv2.dilate(edges, kernel, iterations=1)

    # Contorno de la máscara dentro de la región del bbox. La máscara está en el espacio del
    # lienzo completo, así que extraemos la región del bbox.
    mask_roi = mask_bin[y1:y2, x1:x2]
    mask_u8 = (mask_roi > 127).astype(np.uint8) if mask_roi.max() > 1 else mask_roi.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return 0.0

    # Dibujamos el contorno como una línea de 1px.
    contour_img = np.zeros_like(mask_u8)
    cv2.drawContours(contour_img, contours, -1, 1, 1)

    contour_pixels = contour_img.sum()
    if contour_pixels == 0:
        return 0.0

    # Cuántos píxeles del contorno caen sobre bordes detectados.
    aligned = np.logical_and(contour_img > 0, edge_zone > 0).sum()
    return float(aligned / contour_pixels)


def _remove_border_components(mask: np.ndarray) -> np.ndarray:
    """
    Elimina los componentes conexos de la máscara binaria que tocan cualquier borde de la
    imagen. Así se descartan fragmentos espurios (pelo, fondo, etc.) que llegan hasta el borde,
    sin perder el objeto principal, que normalmente está centrado.

    Si TODOS los componentes tocan algún borde, devuelve la máscara sin cambios para no quedarse
    sin resultado.
    """
    h, w = mask.shape[:2]
    mask_bin = (mask > 127).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(mask_bin)
    # label 0 = fondo; los componentes reales van de 1 a n_labels-1.
    if n_labels <= 1:
        return mask  # Sin componentes, nada que filtrar.

    # Máscara de borde: fila 0, fila H-1, columna 0 y columna W-1.
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[h - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, w - 1] = True

    # Qué etiquetas tocan el borde.
    border_labels = set(np.unique(labels[border_mask]))
    border_labels.discard(0)  # Ignoramos el fondo.

    if len(border_labels) == 0:
        return mask  # Ningún componente toca el borde.

    interior_labels = set(range(1, n_labels)) - border_labels

    if len(interior_labels) == 0:
        print("Todos los componentes tocan bordes — se conserva la máscara completa")
        return mask

    # Reconstruimos la máscara solo con los componentes interiores.
    clean = np.zeros_like(mask_bin)
    for lbl in interior_labels:
        clean[labels == lbl] = 1

    removed_px = mask_bin.sum() - clean.sum()
    print(f"   Bordes: {len(border_labels)} componente(s) eliminado(s), "
          f"{removed_px} píxeles removidos")

    return clean * 255


def _keep_largest_components(mask: np.ndarray, min_ratio: float = 0.10) -> np.ndarray:
    """
    Filtra los fragmentos desconectados de la máscara, conservando solo el componente más
    grande y cualquier otro cuya área sea al menos `min_ratio` del principal. Descarta los
    fragmentos pequeños (ruido, artefactos) que no pertenecen al objeto.
    """
    mask_bin = (mask > 127).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(mask_bin)
    if n_labels <= 2:
        return mask  # 0 o 1 componente, nada que filtrar.

    # Área de cada etiqueta (el índice 0 es el fondo).
    areas = np.bincount(labels.ravel())
    component_areas = areas[1:]  # solo los componentes de primer plano

    max_area = component_areas.max()
    threshold = max_area * min_ratio

    # Etiquetas a conservar (etiqueta = índice + 1 porque la 0 es el fondo).
    keep_labels = {i + 1 for i, area in enumerate(component_areas)
                   if area >= threshold}

    if len(keep_labels) == n_labels - 1:
        return mask  # Todos los componentes son suficientemente grandes.

    # Reconstruimos la máscara limpia.
    clean = np.zeros_like(mask_bin)
    for lbl in keep_labels:
        clean[labels == lbl] = 1

    removed_count = (n_labels - 1) - len(keep_labels)
    removed_px = mask_bin.sum() - clean.sum()
    print(f"   Fragmentos: {removed_count} componente(s) pequeño(s) eliminado(s), "
          f"{removed_px} píxeles descartados (umbral={threshold:.0f}px, "
          f"{min_ratio * 100:.0f}% del principal)")

    return clean * 255


# --- 5. Debug: inspeccionar todas las máscaras candidatas ---

def _debug_masks(results, canvas_bbox: list[int], canvas_size: int):
    """Imprime estadísticas de todas las máscaras para diagnóstico."""
    masks_data = results[0].masks.data.cpu().numpy()
    n_masks = masks_data.shape[0]
    x1, y1, x2, y2 = canvas_bbox
    mH, mW = masks_data.shape[1], masks_data.shape[2]
    sx, sy = mW / canvas_size, mH / canvas_size

    print(f"DEBUG: {n_masks} máscaras generadas, mask_shape=({mH},{mW})")
    for i in range(n_masks):
        m = masks_data[i] > 0.5
        area = m.sum()
        total = mH * mW

        # Cobertura dentro del bbox.
        mx1, my1 = int(x1 * sx), int(y1 * sy)
        mx2, my2 = int(x2 * sx), int(y2 * sy)
        bbox_region = m[my1:my2, mx1:mx2]
        bbox_coverage = bbox_region.sum() / (bbox_region.size + 1e-6) if bbox_region.size > 0 else 0

        print(f"  mask[{i}]: area={area} ({100*area/total:.1f}%), "
              f"bbox_coverage={100*bbox_coverage:.1f}%")

    # Scores de confianza, si están disponibles.
    if results[0].boxes is not None and results[0].boxes.conf is not None:
        confs = results[0].boxes.conf.cpu().numpy()
        for i, c in enumerate(confs):
            print(f"  mask[{i}] conf={c:.3f}")


# --- Pipeline principal: preprocesado -> inferencia -> postprocesado -> RGBA ---

def segment_with_fastsam(
    pil_image: Image.Image,
    bbox: list[int],
    debug: bool = False,
    target_size: int = 1024,
    timings: dict | None = None,
) -> bytes:
    """
    Pipeline completo de segmentación con preprocesado controlado.

    Los pasos son: preprocesar (resize + relleno centrado en lienzo cuadrado), ejecutar FastSAM
    sobre el lienzo (no sobre la imagen cruda), seleccionar la máscara (point prompt -> box
    prompt -> score manual), postprocesar (revertir relleno/escala al tamaño original) y
    componer un RGBA con fondo transparente.

    `pil_image` es una imagen PIL (cualquier modo) y `bbox` es [x1, y1, x2, y2] en píxeles
    absolutos de la imagen original. Con `debug=True` se imprimen estadísticas de todas las
    máscaras. `target_size` es el lado del lienzo cuadrado de trabajo (letterbox): el lienzo
    real se satura en min(target_size, lado_mayor_original), así que subirlo por encima del
    lado mayor de la imagen no cambia la inferencia (útil para medir latencia a distintas
    resoluciones). Si se pasa un dict en `timings`, se rellenan las claves preprocess_ms,
    inference_ms, postprocess_ms y total_ms con la latencia de cada fase.

    Devuelve los bytes de un PNG con fondo transparente.
    """
    w, h = pil_image.size
    rgb = pil_image.convert("RGB")
    img_np = np.array(rgb)

    print(f"Pipeline FastSAM: bbox={bbox} sobre {w}x{h}")

    _local_timings: dict = {} if timings is None else timings
    _t_total = time.perf_counter()

    # 1. Preprocesado.
    _t_pre = time.perf_counter()
    canvas, meta = preprocess_image(img_np, target_size=target_size)
    _local_timings["preprocess_ms"] = (time.perf_counter() - _t_pre) * 1000.0
    print(f"   Lienzo: {meta['canvas_size']}x{meta['canvas_size']}, "
          f"scale={meta['scale']:.3f}, pad=({meta['pad_x']},{meta['pad_y']})")

    # Dibujamos el bbox sobre el original y sobre el lienzo para el debug.
    orig_with_bbox = img_np.copy()
    cv2.rectangle(orig_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), max(2, w // 500))
    cb = _bbox_to_canvas(bbox, meta)
    canvas_with_bbox = canvas.copy()
    cv2.rectangle(canvas_with_bbox, (cb[0], cb[1]), (cb[2], cb[3]), (0, 255, 0), 2)

    debug_imgs = {
        "00_original_bbox": orig_with_bbox,
        "01_canvas_bbox": canvas_with_bbox,
    }

    # 2. Inferencia + selección de máscara.
    # run_fastsam mide la inferencia pura del modelo en timings["inference_ms"]; el resto del
    # tiempo de esta llamada (la selección de máscara) se contabiliza como postprocesado.
    _t_run = time.perf_counter()
    mask_canvas = run_fastsam(canvas, meta, bbox, debug=debug,
                              debug_imgs=debug_imgs, timings=_local_timings)
    _run_total_ms = (time.perf_counter() - _t_run) * 1000.0
    _selection_ms = _run_total_ms - _local_timings.get("inference_ms", 0.0)
    _t_post = time.perf_counter()

    # Fallback: si no se obtuvo máscara, devolvemos un recorte rectangular.
    if mask_canvas is None:
        print("Sin máscara válida — devolviendo recorte rectangular")
        mask_original = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask_original[y1:y2, x1:x2] = 255
    else:
        debug_imgs["10_mask_canvas_selected"] = mask_canvas
        # 3. Postprocesado.
        mask_original = postprocess_mask(mask_canvas, meta)
        debug_imgs["11_mask_original_postproc"] = mask_original

    # Binarizamos por seguridad (INTER_NEAREST debería mantenerla, pero lo aseguramos).
    mask_binary = (mask_original > 127).astype(np.uint8) * 255

    # 3a. Clip estricto al bbox del usuario.
    mask_binary = _clip_mask_to_bbox(mask_binary, bbox)
    debug_imgs["12_mask_after_clip_bbox"] = mask_binary

    # 3b. Eliminar componentes conexos que tocan los bordes.
    mask_binary = _remove_border_components(mask_binary)
    debug_imgs["13_mask_after_remove_border"] = mask_binary

    # 3c. Filtrar fragmentos: conservar solo los componentes grandes.
    mask_binary = _keep_largest_components(mask_binary)
    debug_imgs["14_mask_final"] = mask_binary

    # 3d. Antialiasing del borde: suavizar el dentado del borde binario.
    mask_alpha = _antialias_mask(mask_binary, meta["scale"])
    debug_imgs["14b_mask_antialiased"] = mask_alpha

    # 4. Componer RGBA.
    mask_pil = Image.fromarray(mask_alpha, mode="L")
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    result.paste(rgb, mask=mask_pil)

    # Guardamos un overlay del resultado sobre fondo gris para inspección visual.
    overlay = np.full((h, w, 3), 64, dtype=np.uint8)
    overlay[mask_binary > 0] = img_np[mask_binary > 0]
    debug_imgs["15_result_overlay"] = overlay

    _save_seg_debug("seg", debug_imgs)

    buf = io.BytesIO()
    result.save(buf, format="PNG")

    # postprocesado = selección de máscara (parte de run_fastsam) + revertir el relleno +
    # limpieza + composición RGBA.
    _local_timings["postprocess_ms"] = _selection_ms + (time.perf_counter() - _t_post) * 1000.0
    _local_timings["total_ms"] = (time.perf_counter() - _t_total) * 1000.0
    print(
        f"Latencia: pre={_local_timings['preprocess_ms']:.1f} ms | "
        f"infer={_local_timings.get('inference_ms', 0.0):.1f} ms | "
        f"post={_local_timings['postprocess_ms']:.1f} ms | "
        f"total={_local_timings['total_ms']:.1f} ms "
        f"(lienzo={meta['canvas_size']}px)"
    )

    print(f"Segmentación completada — {w}x{h}")
    return buf.getvalue()
