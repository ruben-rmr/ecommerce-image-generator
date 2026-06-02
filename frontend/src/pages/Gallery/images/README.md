# Imágenes de la galería

Deja aquí las imágenes que quieras mostrar en la página **Galería**
(`/galeria`). Se detectan automáticamente al arrancar/compilar la app.

- Formatos admitidos: `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`, `.gif`
- El **orden** sigue el nombre del archivo (alfabético/numérico). Puedes
  prefijar con números para ordenarlas: `01-marmol.jpg`, `02-madera.jpg`, …
- El **título** se genera a partir del nombre del archivo:
  `fondo-marmol.jpg` → "Fondo Marmol".

Para personalizar el título o la descripción de una imagen concreta, edita el
objeto `META` en [`../galleryData.js`](../galleryData.js), por ejemplo:

```js
const META = {
  'fondo-marmol.jpg': { title: 'Mármol', description: 'Estudio premium' },
}
```
