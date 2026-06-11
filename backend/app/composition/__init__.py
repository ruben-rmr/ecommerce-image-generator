"""
Pipeline de composición local (sin IA generativa).

Hay dos modos:
    - studio: fondo procedural blanco/gris con sombras que asientan el objeto.
    - scene:  composición sobre una imagen de fondo del catálogo, armonizando luz y color.

El objeto segmentado se respeta tal cual. Solo se ajustan el tono y la luminancia
globales con factores acotados. El postprocesado fuerte (descontaminación de halo,
suavizado de bordes) actúa sobre el canal alfa, nunca sobre el RGB interior de la silueta.
"""

from .studio import compose_studio
from .scene import compose_scene

__all__ = ["compose_studio", "compose_scene"]
