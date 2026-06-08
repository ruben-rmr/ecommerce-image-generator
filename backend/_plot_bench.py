"""Grafica los tiempos de segmentacion por resolucion.

Uso:  python _plot_bench.py
Genera 'bench_segmentation.png' en el mismo directorio.
"""
import matplotlib.pyplot as plt

# (resolucion, preprocesado, inferencia, post-procesado, total) en ms
datos = [
    ("640px",  0.4, 148.8,  210.4,  360.1),
    ("960px",  1.4, 384.9,  494.5,  881.7),
    ("1280px", 2.4, 450.4,  636.8, 1090.9),
    ("1920px", 3.0, 443.1,  764.1, 1211.7),
    ("2560px", 3.0, 447.1,  938.0, 1391.5),
    ("3840px", 3.9, 462.3, 1268.2, 1738.6),
]

resoluciones = [d[0] for d in datos]
preprocesado = [d[1] for d in datos]
inferencia   = [d[2] for d in datos]
postproceso  = [d[3] for d in datos]
total        = [d[4] for d in datos]

fig, ax = plt.subplots(figsize=(9, 5))

# Barras apiladas de cada etapa
ax.bar(resoluciones, preprocesado, label="Preprocesado")
ax.bar(resoluciones, inferencia, bottom=preprocesado, label="Inferencia")
base = [p + i for p, i in zip(preprocesado, inferencia)]
ax.bar(resoluciones, postproceso, bottom=base, label="Post-procesado")

# Etiqueta del total encima de cada barra
for x, t in zip(resoluciones, total):
    ax.text(x, t + 20, f"{t:.0f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Resolucion")
ax.set_ylabel("Tiempo (ms)")
ax.set_title("Tiempos de segmentacion por resolucion")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

fig.savefig("bench_segmentation.png", dpi=150)
print("Guardado: bench_segmentation.png")
plt.show()
