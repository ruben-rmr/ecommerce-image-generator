# Backgrounds locales para MODO 2 (Escena comercial)

Estructura:
```
backgrounds/
  <categoria>/
    <nombre>.jpg          # imagen del fondo (jpg/jpeg/png/webp)
    <nombre>.json         # metadatos opcionales (mismo nombre base)
```

Las categorías son simplemente subcarpetas (ej. `oceano/`, `madera/`,
`marmol/`, `cosmetica/`, `rocas/`).

## Esquema de metadatos (JSON, todas las claves opcionales)

```json
{
  "label":           "Mármol pulido",
  "ground_y":        0.72,
  "light_dir":       [0.4, -0.6],
  "reflective":      true,
  "reflective_type": "glossy"
}
```

- `label`: texto mostrado en la UI. Por defecto, el nombre del archivo.
- `ground_y`: posición relativa (0..1) de la línea del suelo/horizonte donde se
  apoyarán los pies del producto. Si se omite, se detecta automáticamente.
- `light_dir`: vector 2D (x, y) que apunta hacia el origen de la luz. El eje y
  va hacia abajo en imágenes, así que `[0.4, -0.6]` es luz arriba-derecha. Si
  se omite, se estima por la región más brillante del fondo.
- `reflective`: si `true`, se genera reflejo vertical bajo el producto.
- `reflective_type`: `"glossy"` (espejo casi limpio) o `"matte"` (reflejo
  difuso, blur horizontal mayor).

Las imágenes se sirven mediante:
- `GET /backgrounds/`                            → catálogo agrupado por categoría
- `GET /backgrounds/thumb/<categoria>/<nombre>`  → thumbnail 256 px
- `GET /backgrounds/full/<categoria>/<nombre>`   → imagen completa
