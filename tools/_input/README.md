# Entrada para los scripts de figuras

Coloca aquí el **PNG del producto con fondo transparente** (RGBA), es decir, la
salida del paso **Segmentar** de la aplicación (`segmentado.png`).

Por defecto los scripts buscan `tools/_input/producto.png`.

> ⚠️ No sirve un PNG con fondo gris/opaco: la sombra y la armonización se calculan
> a partir del canal alfa. Necesita ser el recorte transparente.

## Cómo generarlo
1. Arranca backend (`uvicorn main:app --reload`) y frontend (`npm run dev`).
2. En **Profesionalizar**, sube la foto, segmenta (auto o bbox manual) y pulsa
   **Descargar PNG**.
3. Guarda ese fichero aquí como `producto.png`.