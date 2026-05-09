from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import io
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from segmentation import segment_with_fastsam
from utils import detectar_bbox_canny
from composition import compose_studio, compose_scene
from composition.catalog import BackgroundCatalog


BACKGROUNDS_DIR = Path(__file__).parent / "backgrounds"
catalog = BackgroundCatalog(BACKGROUNDS_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-cargar el modelo de segmentación.
    from segmentation import _get_model
    _get_model()
    # Escanear catálogo de fondos locales.
    catalog.scan()
    print("🚀 Servidor iniciado (FastSAM local + composición local)")
    yield
    print("🛑 Servidor apagándose...")


app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------
def _read_image(contents: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(contents))
    image = ImageOps.exif_transpose(image)
    return image


def _parse_optional_position(raw: str | None) -> tuple[float, float] | None:
    if raw is None or raw == "":
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            return float(data[0]), float(data[1])
    except Exception:
        pass
    return None


def _parse_canvas_size(raw: str | None) -> tuple[int, int] | None:
    if raw is None or raw == "":
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            w, h = int(data[0]), int(data[1])
            if w > 0 and h > 0:
                return (w, h)
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------
# RUTAS BÁSICAS
# ----------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Backend TFG con FastSAM + composición local",
        "modes": ["studio", "scene"],
    }


# ----------------------------------------------------------------------
# SEGMENTACIÓN — sin cambios funcionales
# ----------------------------------------------------------------------
def _resize_for_segmentation(image: Image.Image, max_side: int | None) -> Image.Image:
    """Redimensiona manteniendo aspect ratio si algún lado supera max_side."""
    if max_side is None:
        return image
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"🔲 Resize: {w}×{h} → {new_w}×{new_h} (max_side={max_side})")
    return image.resize((new_w, new_h), Image.LANCZOS)


@app.post("/segment_bbox/")
async def segment_image_bbox(
    file: UploadFile = File(...),
    bbox: str = Form(...),
    max_side: int | None = Form(None),
):
    """Segmentación con bounding box manual relativo (0-1)."""
    try:
        try:
            coords = json.loads(bbox)
            if len(coords) != 4:
                raise ValueError
            rel_x1, rel_y1, rel_x2, rel_y2 = [float(c) for c in coords]
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400,
                                detail="bbox debe ser un JSON array con 4 coordenadas: [x1, y1, x2, y2]")

        contents = await file.read()
        image = _read_image(contents)
        image = _resize_for_segmentation(image, max_side)
        w, h = image.size

        px1 = max(0, min(int(rel_x1 * w), w - 1))
        py1 = max(0, min(int(rel_y1 * h), h - 1))
        px2 = max(0, min(int(rel_x2 * w), w - 1))
        py2 = max(0, min(int(rel_y2 * h), h - 1))
        bx1, bx2 = min(px1, px2), max(px1, px2)
        by1, by2 = min(py1, py2), max(py1, py2)

        print(f"📦 Bbox manual: rel=[{rel_x1:.3f},{rel_y1:.3f},{rel_x2:.3f},{rel_y2:.3f}] "
              f"→ px=[{bx1},{by1},{bx2},{by2}] sobre {w}×{h}")

        result_png = await asyncio.to_thread(segment_with_fastsam, image, [bx1, by1, bx2, by2])
        return Response(content=result_png, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en segmentación bbox: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment_auto/")
async def segment_image_auto(
    file: UploadFile = File(...),
    max_side: int | None = Form(None),
):
    """Segmentación automática (Canny → bbox → FastSAM)."""
    try:
        contents = await file.read()
        image = _read_image(contents)
        image = _resize_for_segmentation(image, max_side)
        bbox = await asyncio.to_thread(detectar_bbox_canny, image)
        print(f"📦 Bbox automático (Canny): {bbox} sobre {image.size[0]}×{image.size[1]}")
        result_png = await asyncio.to_thread(segment_with_fastsam, image, bbox)
        return Response(content=result_png, media_type="image/png")

    except Exception as e:
        print(f"❌ Error en segmentación auto: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# COMPOSICIÓN LOCAL — MODO 1 (estudio) y MODO 2 (escena)
# ----------------------------------------------------------------------
@app.post("/compose/studio/")
async def compose_studio_endpoint(
    file: UploadFile = File(...),
    style: str = Form("white"),
    position: str | None = Form(None),
    scale: float | None = Form(None),
    canvas: str | None = Form(None),
):
    """
    Compone el PNG segmentado sobre un fondo procedural de estudio.
    - style: 'white' | 'soft_gray'
    - position: JSON [rel_cx, rel_y_feet] opcional (override manual)
    - scale: ratio altura silueta / altura canvas (override)
    - canvas: JSON [W, H] opcional. Por defecto 1024x1024.
    """
    try:
        png_bytes = await file.read()
        manual_position = _parse_optional_position(position)
        canvas_size = _parse_canvas_size(canvas) or (1024, 1024)
        manual_scale = float(scale) if scale is not None else None

        result_png = await asyncio.to_thread(
            compose_studio,
            png_bytes,
            style,
            canvas_size,
            manual_position,
            manual_scale,
        )
        return Response(content=result_png, media_type="image/png")

    except Exception as e:
        print(f"❌ Error en compose/studio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compose/scene/")
async def compose_scene_endpoint(
    file: UploadFile = File(...),
    background_id: str = Form(...),
    position: str | None = Form(None),
    scale: float | None = Form(None),
    harmonize_strength: float = Form(0.45),
    canvas: str | None = Form(None),
):
    """
    Compone el PNG segmentado sobre un background local.
    - background_id: '<categoria>/<nombre>' (ver GET /backgrounds/)
    - position: JSON [rel_cx, rel_y_feet] opcional
    - scale: ratio altura silueta / altura canvas
    - harmonize_strength: 0..1 (a/b LAB transfer)
    - canvas: JSON [W, H] opcional. Por defecto = tamaño del fondo.
    """
    try:
        entry = catalog.get(background_id)
        if entry is None:
            raise HTTPException(status_code=404,
                                detail=f"background_id '{background_id}' no existe")

        metadata = {
            "ground_y":        entry.ground_y,
            "light_dir":       list(entry.light_dir) if entry.light_dir else None,
            "reflective":      entry.reflective,
            "reflective_type": entry.reflective_type,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        png_bytes = await file.read()
        manual_position = _parse_optional_position(position)
        canvas_size = _parse_canvas_size(canvas)
        manual_scale = float(scale) if scale is not None else None

        result_png = await asyncio.to_thread(
            compose_scene,
            png_bytes,
            entry.path,
            metadata,
            canvas_size,
            manual_position,
            manual_scale,
            float(harmonize_strength),
        )
        return Response(content=result_png, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en compose/scene: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# CATÁLOGO DE BACKGROUNDS
# ----------------------------------------------------------------------
@app.get("/backgrounds/")
def list_backgrounds():
    """Devuelve { categoria: [ { id, name, label, thumb_url, full_url, reflective, ... } ] }."""
    return JSONResponse(content=catalog.list_grouped())


@app.post("/backgrounds/rescan/")
def rescan_backgrounds():
    """Re-escanea la carpeta de backgrounds (útil tras añadir imágenes en caliente)."""
    catalog.scan()
    return {"ok": True, "categories": list(catalog.list_grouped().keys())}


@app.get("/backgrounds/full/{category}/{name}")
def get_background_full(category: str, name: str):
    path = catalog.get_path(category, name)
    if not path:
        raise HTTPException(status_code=404, detail="background no encontrado")
    data = path.read_bytes()
    media = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return Response(content=data, media_type=media)


@app.get("/backgrounds/thumb/{category}/{name}")
def get_background_thumb(category: str, name: str):
    path = catalog.get_path(category, name)
    if not path:
        raise HTTPException(status_code=404, detail="background no encontrado")
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((256, 256), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
