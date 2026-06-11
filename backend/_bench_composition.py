"""
Benchmark de latencia de COMPOSICIÓN por etapas (modo estudio y modo escena).

A diferencia del de segmentación, las funciones `compose_studio` / `compose_scene`
no exponen un parámetro `timings`, así que este script **replica fielmente el orden
de operaciones de ambos pipelines** (los mismos submódulos de `app.composition`)
cronometrando cada etapa por separado con `time.perf_counter`. El total reportado es
la suma exacta de las etapas medidas, de modo que la tabla cuadra.

Valor diagnóstico: identifica qué etapa domina el coste del modo escena (la candidata
natural a optimizar en trabajo futuro).

Uso (desde backend/, con el venv activado):
    python _bench_composition.py                       # objeto=segmentado.png, fondo=escena_1.png
    python _bench_composition.py --object segmentado.png --background escena_1.png
    python _bench_composition.py --canvas 1024 --repeats 10
    python _bench_composition.py --reflective           # incluye la etapa de reflejo

El objeto debe ser un PNG RGBA (recorte segmentado) y el fondo un JPG/PNG, ambos en
la misma carpeta que este script. Imprime dos tablas (Operación | Tiempo (ms)) con la
media de cada etapa sobre N repeticiones, tras un warm-up que no se contabiliza.
"""
import argparse
import os
import time

import numpy as np

from app.composition.io_utils import (
    png_bytes_to_rgba, load_background,
    alpha_blend, multiply_shadow, compute_object_footprint,
)
from app.composition.edges import prepare_object
from app.composition.placement import (
    auto_scale, place_on_scene, detect_ground_y, background_roi_below_object,
)
from app.composition.harmonization import (
    harmonize_lab, brightness_contrast_match, atmospheric_blend,
)
from app.composition.relighting import (
    estimate_light_dir, apply_directional_relight, apply_studio_keyfill,
)
from app.composition.shadows import contact_shadow, cast_shadow
from app.composition.reflections import make_reflection, reflection_top_left
from app.composition.studio import _make_canvas, _paste_object
from app.composition.scene import _paste_rgba

HERE = os.path.dirname(os.path.abspath(__file__))


class _Timer:
    """Acumula tiempos por etiqueta a lo largo de varias repeticiones (ms)."""
    def __init__(self):
        self.acc = {}        # label -> list[ms]
        self.order = []      # conserva el orden de aparición

    def time(self, label, fn):
        t0 = time.perf_counter()
        out = fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if label not in self.acc:
            self.acc[label] = []
            self.order.append(label)
        self.acc[label].append(dt)
        return out

    def means(self):
        return [(lbl, float(np.mean(self.acc[lbl]))) for lbl in self.order]


# --- Modo estudio ---
def run_studio(png_bytes, canvas_size, style, tm):
    """Replica compose_studio() midiendo cada etapa de la tabla."""
    W, H = canvas_size

    # "Pegado del objeto" agrupa decodificación + limpieza + escalado + keyfill + pegado.
    def _prep_object():
        rgba = png_bytes_to_rgba(png_bytes)
        rgba = prepare_object(rgba)
        obj_resized, top_left = auto_scale(rgba, canvas_size, 0.62)
        return obj_resized, top_left

    obj_resized, top_left = _prep_object()  # (no medido aún; medimos pegado completo abajo)

    canvas = tm.time("Construcción del lienzo",
                     lambda: _make_canvas(canvas_size, style))

    def _drop():
        drop = cast_shadow(canvas_size, obj_resized[..., 3], top_left,
                           light_dir=(-0.6, -0.6), length=0.45, squash=0.20,
                           fade=0.5, sigma_contact=4.0, sigma_tip=20.0, intensity=0.55)
        return multiply_shadow(canvas, drop, color=(0, 0, 0), opacity=1.0)
    canvas = tm.time("Sombra proyectada", _drop)

    def _paste():
        obj_lit = apply_studio_keyfill(obj_resized, amplitude=6.0, mix=0.25)
        return _paste_object(canvas, obj_lit, top_left)
    canvas = tm.time("Pegado del objeto", _paste)

    def _contact():
        c = contact_shadow(canvas_size, obj_resized[..., 3], top_left,
                           intensity=0.55, sigma=3.0, band_ratio=0.08)
        return multiply_shadow(canvas, c, color=(0, 0, 0), opacity=1.0)
    canvas = tm.time("Sombra de contacto", _contact)
    return canvas


# --- Modo escena ---
def run_scene(png_bytes, bg_path, canvas_size, reflective, tm):
    """Replica compose_scene() midiendo cada etapa de la tabla."""
    bg = tm.time("Carga del fondo",
                 lambda: load_background(bg_path, target_size=canvas_size))
    H, W = bg.shape[:2]
    cs = (W, H)

    # 1) Limpieza del objeto (descontaminación de halo + micro-feather).
    def _clean():
        rgba = png_bytes_to_rgba(png_bytes)
        return prepare_object(rgba)
    rgba = tm.time("Limpieza del objeto", _clean)

    # 2) Posicionamiento (detección de suelo + escalado + colocación).
    state = {}
    def _place():
        ground_y = float(detect_ground_y(bg))
        obj_placed, top_left = place_on_scene(rgba, cs, ground_y_rel=ground_y,
                                              target_height_ratio=0.40)
        state["obj"] = obj_placed
        state["tl"] = top_left
        return obj_placed
    tm.time("Posicionamiento", _place)
    obj_placed, top_left = state["obj"], state["tl"]

    # 3-4) Armonización LAB (transfer Reinhard parcial + ajuste de brillo).
    def _harmonize():
        footprint = compute_object_footprint(obj_placed[..., 3])
        bg_roi = background_roi_below_object(bg, top_left, obj_placed.shape[1::-1],
                                             footprint, expand=0.5)
        oh = harmonize_lab(obj_placed, bg_roi, strength=0.45)
        oh = brightness_contrast_match(oh, bg_roi, strength=0.30)
        state["footprint"] = footprint
        return oh
    obj_harm = tm.time("Armonización LAB", _harmonize)

    # 5) Reiluminación direccional (estimación de luz + gradiente L sobre la silueta).
    def _relight():
        light_dir = estimate_light_dir(bg)
        state["light"] = light_dir
        return apply_directional_relight(obj_harm, light_dir, amplitude=5.0, mix=0.20)
    obj_harm = tm.time("Reiluminación direccional", _relight)
    light_dir = state["light"]

    # 6) Fusión atmosférica en el anillo del borde.
    obj_harm = tm.time("Fusión atmosférica",
                       lambda: atmospheric_blend(bg, top_left, obj_harm,
                                                 ring_thickness=3, bg_blur_sigma=3.0, mix=0.45))

    # 7) Sombra proyectada (warp sobre buffer reducido + blend).
    canvas_holder = {"c": None}
    def _drop():
        canvas = bg.copy()
        drop = cast_shadow(cs, obj_harm[..., 3], top_left,
                           light_dir=light_dir, length=0.55, squash=0.28,
                           fade=0.5, sigma_contact=5.0, sigma_tip=26.0, intensity=0.55)
        canvas_holder["c"] = multiply_shadow(canvas, drop, color=(0, 0, 0), opacity=1.0)
        return canvas_holder["c"]
    canvas = tm.time("Sombra proyectada", _drop)

    # Reflejo (opcional).
    if reflective:
        def _reflection():
            reflection = make_reflection(obj_harm, fade=0.30, blur_sigma=4.0)
            _, _, _, fy_max = state["footprint"]
            rx, ry = reflection_top_left(top_left, obj_harm.shape[1::-1], feet_y_local=fy_max)
            return _paste_rgba(canvas_holder["c"], reflection, (rx, ry))
        canvas = tm.time("Reflejo", _reflection)
        canvas_holder["c"] = canvas

    # Pegado del objeto.
    canvas = tm.time("Pegado del objeto",
                     lambda: _paste_rgba(canvas_holder["c"], obj_harm, top_left))

    # Sombra de contacto (nítida, encima).
    def _contact():
        c = contact_shadow(cs, obj_harm[..., 3], top_left,
                           intensity=0.55, sigma=3.0, band_ratio=0.08)
        return multiply_shadow(canvas, c, color=(0, 0, 0), opacity=1.0)
    canvas = tm.time("Sombra de contacto", _contact)
    return canvas


def _print_table(title, pairs):
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)
    print(f"{'Operación':<34} | {'Tiempo (ms)':>15}")
    print("-" * 56)
    total = 0.0
    for lbl, ms in pairs:
        print(f"{lbl:<34} | {ms:>15.2f}")
        total += ms
    print("-" * 56)
    print(f"{'TOTAL':<34} | {total:>15.2f}")
    print("=" * 56)


# Resoluciones de trabajo a medir = lado del lienzo cuadrado en px.
RESOLUTIONS = [640, 960, 1280, 1920, 2560, 3840]


def _print_matrix(title, labels, by_res, resolutions):
    """Tabla operación × resolución. `by_res[res]` es dict label->ms."""
    print("\n" + "=" * (36 + 11 * len(resolutions)))
    print(title)
    print("=" * (36 + 11 * len(resolutions)))
    hdr = f"{'Operación':<34}" + "".join(f"|{str(r) + ' px':>10}" for r in resolutions)
    print(hdr)
    print("-" * len(hdr))
    totals = {r: 0.0 for r in resolutions}
    for lbl in labels:
        row = f"{lbl:<34}"
        for r in resolutions:
            ms = by_res[r].get(lbl, float("nan"))
            row += f"|{ms:>10.2f}"
            if ms == ms:  # no NaN
                totals[r] += ms
        print(row)
    print("-" * len(hdr))
    trow = f"{'TOTAL':<34}" + "".join(f"|{totals[r]:>10.2f}" for r in resolutions)
    print(trow)
    print("=" * len(hdr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="segmentado.png",
                    help="PNG RGBA del objeto segmentado (en esta carpeta).")
    ap.add_argument("--background", default="escena_1.png",
                    help="Imagen de fondo para el modo escena (en esta carpeta).")
    ap.add_argument("--canvas", type=int, default=None,
                    help="Mide un único lienzo cuadrado de este lado (px). "
                         "Si se omite, barre RESOLUTIONS (640..3840).")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Repeticiones por etapa (media). Por defecto 5.")
    ap.add_argument("--style", default="white", choices=("white", "soft_gray"),
                    help="Estilo del fondo de estudio.")
    ap.add_argument("--reflective", action="store_true",
                    help="Incluye la etapa de reflejo en el modo escena.")
    args = ap.parse_args()

    obj_path = args.object if os.path.isabs(args.object) else os.path.join(HERE, args.object)
    bg_path = args.background if os.path.isabs(args.background) else os.path.join(HERE, args.background)
    resolutions = [args.canvas] if args.canvas else RESOLUTIONS

    if not os.path.isfile(obj_path):
        raise SystemExit(f"No existe el objeto: {obj_path}")
    if not os.path.isfile(bg_path):
        raise SystemExit(f"No existe el fondo: {bg_path}")

    with open(obj_path, "rb") as f:
        png_bytes = f.read()

    print(f"Objeto : {obj_path}")
    print(f"Fondo  : {bg_path}")
    print(f"Resoluciones (lado del lienzo): {resolutions}  |  repeticiones: {args.repeats}")
    print("AVISO: las resoluciones altas (>=2560 px) tardan ~decenas de segundos por")
    print("iteración (sombra de contacto). Usa --canvas N para medir una sola.", flush=True)

    studio_by_res, scene_by_res = {}, {}
    studio_labels, scene_labels = [], []
    n = len(resolutions)
    for i, side in enumerate(resolutions, 1):
        canvas_size = (side, side)
        print(f"\n[{i}/{n}] Lienzo {side}x{side}  (calentando...)", flush=True)

        # Warm-up por resolución (no contabilizado): calienta cachés de OpenCV/NumPy.
        warm = _Timer()
        run_studio(png_bytes, canvas_size, args.style, warm)
        run_scene(png_bytes, bg_path, canvas_size, args.reflective, warm)

        tm_studio = _Timer()
        for r in range(args.repeats):
            t0 = time.perf_counter()
            run_studio(png_bytes, canvas_size, args.style, tm_studio)
            print(f"    estudio  rep {r + 1}/{args.repeats}: {(time.perf_counter() - t0) * 1000:8.0f} ms", flush=True)
        tm_scene = _Timer()
        for r in range(args.repeats):
            t0 = time.perf_counter()
            run_scene(png_bytes, bg_path, canvas_size, args.reflective, tm_scene)
            print(f"    escena   rep {r + 1}/{args.repeats}: {(time.perf_counter() - t0) * 1000:8.0f} ms", flush=True)

        studio_by_res[side] = dict(tm_studio.means())
        scene_by_res[side] = dict(tm_scene.means())
        studio_labels = [lbl for lbl, _ in tm_studio.means()]
        scene_labels = [lbl for lbl, _ in tm_scene.means()]

    if len(resolutions) == 1:
        side = resolutions[0]
        _print_table(f"MODO ESTUDIO  ({side}x{side}, estilo={args.style}, media de {args.repeats})",
                     list(studio_by_res[side].items()))
        _print_table(f"MODO ESCENA   ({side}x{side}, media de {args.repeats})",
                     list(scene_by_res[side].items()))
    else:
        _print_matrix(f"MODO ESTUDIO  (estilo={args.style}, media de {args.repeats}, ms)",
                      studio_labels, studio_by_res, resolutions)
        _print_matrix(f"MODO ESCENA   (media de {args.repeats}, ms)",
                      scene_labels, scene_by_res, resolutions)
    if not args.reflective:
        print("\nNota: el reflejo solo se mide con --reflective (la etapa es opcional,")
        print("se dispara cuando el fondo está marcado como 'reflective' en el catálogo).")


if __name__ == "__main__":
    main()
