"""
Perfilado rápido de las funciones de sombra (ms por llamada) sobre una silueta sintética.
Compara cast_shadow, fake_ambient_occlusion y contact_shadow para ver cuál domina el coste.
Ejecutar desde backend/ con el venv activado: python _prof.py
"""
import time
import numpy as np
from app.composition import shadows

oh, ow = 600, 160
alpha = np.zeros((oh, ow), np.uint8)
alpha[:, 40:120] = 255
top_left = (450, 400)

# Calentamiento (no se contabiliza)
shadows.cast_shadow((1024, 1024), alpha, top_left)

N = 20
t0 = time.perf_counter()
for _ in range(N):
    shadows.cast_shadow((1024, 1024), alpha, top_left, light_dir=(-0.6, -0.6),
                        length=0.45, squash=0.20, fade=0.5,
                        sigma_contact=4.0, sigma_tip=20.0, intensity=0.55)
print(f"cast_shadow: {(time.perf_counter()-t0)/N*1000:.1f} ms/call")

t0 = time.perf_counter()
for _ in range(N):
    shadows.fake_ambient_occlusion((1024, 1024), alpha, top_left, radius=30, intensity=0.15)
print(f"fake_ambient_occlusion: {(time.perf_counter()-t0)/N*1000:.1f} ms/call")

t0 = time.perf_counter()
for _ in range(N):
    shadows.contact_shadow((1024, 1024), alpha, top_left)
print(f"contact_shadow: {(time.perf_counter()-t0)/N*1000:.1f} ms/call")
