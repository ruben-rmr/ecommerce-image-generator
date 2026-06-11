"""
Prueba rápida de compose_studio sin levantar el servidor ni cargar FastSAM.

Genera un objeto RGBA sintético, lo compone a dos escalas y guarda el resultado, imprimiendo
de paso el tamaño del PNG y la latencia. Sirve para detectar regresiones obvias en el modo
estudio. Ejecutar desde backend/ con el venv activado: python _smoke_studio.py
"""
import io
import time
import numpy as np
from PIL import Image
from app.composition.studio import compose_studio

# Objeto RGBA sintético (botella alta/estrecha) -> PNG bytes.
oh, ow = 600, 160
rgba = np.zeros((oh, ow, 4), np.uint8)
rgba[:, 40:120, :3] = (180, 60, 60)
rgba[:, 40:120, 3] = 255
buf = io.BytesIO()
Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
png = buf.getvalue()

for scale in (1.0, 1.4):
    t0 = time.perf_counter()
    out = compose_studio(png, style="soft_gray", manual_scale=scale)
    ms = (time.perf_counter() - t0) * 1000.0
    Image.open(io.BytesIO(out)).save(f"_out_studio_{scale}.png")
    print(f"scale={scale}: {len(out)} bytes, {ms:.0f} ms -> _out_studio_{scale}.png")

print("compose_studio OK")
