from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import traceback
from PIL import Image, ImageOps
import io

# Importamos nuestros módulos locales
from segmentation import segment_with_sam_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Servidor iniciado (segmentación vía Replicate API)")
    yield
    print("🛑 Servidor apagándose...")

app = FastAPI(lifespan=lifespan)

# --- CONFIGURACIÓN CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend TFG con SAM2 (Replicate) funcionando"}


# ==========================================================
# SEGMENTACIÓN CON BOUNDING BOX (SAM2 vía Replicate)
# ==========================================================
@app.post("/segment_bbox/")
async def segment_image_bbox(
    file: UploadFile = File(...),
    bbox: str        = Form(...),   # JSON: [relX1, relY1, relX2, relY2] en 0-1
):
    """
    Recibe imagen + bounding box relativo (0-1) dibujado por el usuario.
    Devuelve un PNG con fondo transparente.
    """
    try:
        # ── Parsear bbox ────────────────────────────────────────────────────
        try:
            coords = json.loads(bbox)
            if len(coords) != 4:
                raise ValueError
            rel_x1, rel_y1, rel_x2, rel_y2 = [float(c) for c in coords]
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="bbox debe ser un JSON array con 4 coordenadas: [x1, y1, x2, y2]",
            )

        contents = await file.read()

        # ── Obtener dimensiones reales de la imagen ─────────────────────────
        image = Image.open(io.BytesIO(contents))
        image = ImageOps.exif_transpose(image)
        w, h = image.size

        # ── Convertir coordenadas relativas (0-1) a píxeles ────────────────
        px1 = max(0, min(int(rel_x1 * w), w - 1))
        py1 = max(0, min(int(rel_y1 * h), h - 1))
        px2 = max(0, min(int(rel_x2 * w), w - 1))
        py2 = max(0, min(int(rel_y2 * h), h - 1))

        bx1, bx2 = min(px1, px2), max(px1, px2)
        by1, by2 = min(py1, py2), max(py1, py2)

        print(
            f"📦 Bbox: rel=[{rel_x1:.3f},{rel_y1:.3f},{rel_x2:.3f},{rel_y2:.3f}] "
            f"→ px=[{bx1},{by1},{bx2},{by2}] sobre {w}×{h}"
        )

        # ── Re-codificar la imagen corregida (EXIF) como PNG en bytes ──────
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        clean_bytes = buf.getvalue()

        # ── Segmentar vía Replicate ─────────────────────────────────────────
        result_png = await segment_with_sam_api(clean_bytes, [bx1, by1, bx2, by2])

        print("✅ Segmentación completada.")
        return Response(content=result_png, media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en segmentación: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
