"""
Prueba rápida de cast_shadow (sombra proyectada) sobre un objeto sintético alto y estrecho,
que es el peor caso de recorte. Comprueba que la sombra no se recorta contra la caja del
objeto: imprime el rango de filas/columnas con sombra, la opacidad en la base y en la punta, y
el ancho proyectado. Ejecutar desde backend/ con el venv activado: python _smoke_cast.py
"""
import numpy as np
from app.composition.shadows import cast_shadow

# Objeto sintético alto/estrecho (peor caso de recorte): barra vertical.
W, H = 1024, 1024
ow, oh = 120, 600
alpha = np.zeros((oh, ow), np.uint8)
alpha[:, 30:90] = 255
top_left = (450, 400)  # base alrededor de y=1000

s = cast_shadow((W, H), alpha, top_left, light_dir=(-0.8, -0.5),
                length=0.7, squash=0.25, fade=0.5,
                sigma_contact=5.0, sigma_tip=26.0, intensity=0.55)

print("dtype", s.dtype, "shape", s.shape,
      "range", round(float(s.min()), 3), round(float(s.max()), 3))

ys, xs = np.where(s > 0.02)
print("filas con sombra:", int(ys.min()), "->", int(ys.max()))
print("cols con sombra:", int(xs.min()), "->", int(xs.max()), "(canvas 0..1023)")
base_row = s[990:1005].max()
tip_row = s[int(ys.min()):int(ys.min()) + 15].max()
print("opacidad base ~", round(float(base_row), 3),
      "| opacidad punta ~", round(float(tip_row), 3))
print("ancho proyectado:", int(xs.max() - xs.min()), "px (caja objeto era 120px)")
