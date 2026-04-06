import replicate
import httpx
import asyncio
import base64
import io
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Modelo de Replicate (SAM2)
# Ajusta el ID si encuentras una versión más estable o actualizada.
# ─────────────────────────────────────────────────────────────────────────────
REPLICATE_MODEL = "meta/sam-2"


async def segment_with_sam_api(image_bytes: bytes, bbox: list[int]) -> bytes:
    """
    Envía la imagen y el bounding box a SAM2 en Replicate.
    Devuelve los bytes de un PNG con fondo transparente.

    Args:
        image_bytes: Imagen original en bytes (JPEG/PNG).
        bbox: Coordenadas absolutas en píxeles [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    print(f"📡 Enviando a Replicate ({REPLICATE_MODEL})  bbox=[{x1},{y1},{x2},{y2}]")

    # Codificar imagen como data-URI para enviarla a Replicate
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:application/octet-stream;base64,{b64}"

    # Llamada a Replicate (síncrona internamente, la envolvemos en un hilo)
    output = await asyncio.to_thread(
        replicate.run,
        REPLICATE_MODEL,
        input={
            "image": data_uri,
            "box": f"{x1},{y1},{x2},{y2}",
        },
    )

    # La respuesta puede ser una URL (str) o una lista de URLs
    if isinstance(output, list):
        mask_url = str(output[0])
    else:
        mask_url = str(output)

    print(f"✅ Máscara recibida: {mask_url[:120]}...")

    # Descargar la máscara generada
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(mask_url)
        resp.raise_for_status()
        mask_bytes = resp.content

    # ── Composición: aplicar máscara sobre la imagen original ────────────────
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Asegurar que la máscara coincida en tamaño con la original
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.LANCZOS)

    # Crear imagen RGBA (objeto recortado con fondo transparente)
    result = Image.new("RGBA", original.size, (0, 0, 0, 0))
    result.paste(original, mask=mask)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print(f"✅ Segmentación completada — {original.size[0]}×{original.size[1]}")
    return buf.getvalue()
