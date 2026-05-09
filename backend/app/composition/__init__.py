"""
Local composition pipeline (no generative AI).

Modes:
    - studio: procedural white/gray background + grounding shadows
    - scene:  composite onto local background image with light harmonization

The segmented object is preserved exactly. Only global tone/luminance is
adjusted with bounded factors. Heavy post-processing (decontamination,
feathering) is applied to the alpha channel, never to the RGB content
inside the object's silhouette.
"""

from .studio import compose_studio
from .scene import compose_scene

__all__ = ["compose_studio", "compose_scene"]
