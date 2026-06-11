"""
Benchmark de latencia de segmentación por fases (pre / inferencia / post).

Mide cómo escala cada fase del pipeline FastSAM al variar la **resolución de
trabajo** = lado mayor de la imagen de entrada. El letterbox satura el lienzo
en target_size=1024 px (valor por defecto del pipeline), así que la hipótesis a
contrastar es:

  - El tiempo de INFERENCIA crece hasta que el lado mayor alcanza 1024 px y
    después se estabiliza (el canvas ya no crece, queda fijo en 1024×1024).
  - El PREPROCESADO y el POSTPROCESADO sí crecen con la resolución de trabajo,
    porque operan sobre el array a tamaño original (resize de entrada y reversión
    del padding/escala a la salida).

Uso (desde backend/, con el venv activado):
    python _bench_segmentation.py                  # imagen sintética
    python _bench_segmentation.py --image foto.jpg # imagen real
    python _bench_segmentation.py --repeats 10

Imprime una tabla con la media de cada fase sobre N repeticiones (tras 1 warm-up
que excluye el coste de carga perezosa del modelo).
"""
import argparse
import numpy as np
from PIL import Image

from app.segmentation import segment_with_fastsam

# Resoluciones de trabajo a medir (lado mayor en px)
RESOLUTIONS = [640, 960, 1280, 1920, 2560]


def _build_synthetic(side: int) -> Image.Image:
    """Imagen sintética cuadrada: un 'producto' coloreado y texturado sobre
    fondo gris claro, para que FastSAM tenga bordes que segmentar."""
    rng = np.random.default_rng(42)
    img = np.full((side, side, 3), 210, dtype=np.uint8)  # fondo gris claro
    m = side // 5
    obj = img[m:side - m, m:side - m]
    # gradiente + ruido suave para dar textura al objeto
    yy = np.linspace(60, 200, obj.shape[0])[:, None]
    obj[:] = np.clip(yy + rng.normal(0, 8, obj.shape), 0, 255).astype(np.uint8)
    obj[..., 0] = np.clip(obj[..., 0].astype(int) + 30, 0, 255)  # tinte rojizo
    img[m:side - m, m:side - m] = obj
    return Image.fromarray(img, mode="RGB")


def _resize_long_side(pil_img: Image.Image, target_long: int) -> Image.Image:
    w, h = pil_img.size
    scale = target_long / max(w, h)
    return pil_img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                          Image.LANCZOS)


def _bbox_for(pil_img: Image.Image) -> list[int]:
    """Bbox absoluto cubriendo el ~60% central de la imagen."""
    w, h = pil_img.size
    return [int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="Ruta a una imagen real. Si se omite, usa una sintética.")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeticiones por resolución (media). Por defecto 5.")
    args = ap.parse_args()

    if args.image:
        base = Image.open(args.image).convert("RGB")
        print(f"Imagen base: {args.image} ({base.size[0]}x{base.size[1]})")
    else:
        base = _build_synthetic(2560)
        print("Imagen base: sintética 2560x2560")

    print(f"Repeticiones por resolución: {args.repeats}\n")

    # Warm-up global: fuerza la carga perezosa del modelo (no se contabiliza).
    warm = _resize_long_side(base, 640)
    segment_with_fastsam(warm, _bbox_for(warm))

    rows = []
    for res in RESOLUTIONS:
        img = _resize_long_side(base, res)
        bbox = _bbox_for(img)

        acc = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": [],
               "total_ms": []}
        for _ in range(args.repeats):
            t = {}
            segment_with_fastsam(img, bbox, timings=t)
            for k in acc:
                acc[k].append(t.get(k, float("nan")))

        row = {k: float(np.mean(v)) for k, v in acc.items()}
        row["res"] = res
        rows.append(row)

    # Tabla de resultados.
    print("\n" + "=" * 78)
    print("RESULTADOS (media de {} repeticiones, ms)".format(args.repeats))
    print("=" * 78)
    hdr = f"{'Lado mayor':>11} | {'Pre':>9} | {'Inferencia':>11} | {'Post':>9} | {'Total':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['res']:>9} px | {r['preprocess_ms']:>9.1f} | "
              f"{r['inference_ms']:>11.1f} | {r['postprocess_ms']:>9.1f} | "
              f"{r['total_ms']:>9.1f}")
    print("=" * 78)
    print("Nota: el lienzo (canvas) se satura en 1024 px; resoluciones de trabajo")
    print(">= 1024 comparten el mismo coste de inferencia (hipótesis).")


if __name__ == "__main__":
    main()
