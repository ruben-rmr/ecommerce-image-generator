# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visión general

App web para **segmentación automática de productos** y **composición sobre fondos personalizados**, 100% local sin APIs externas de pago. El usuario sube una foto, delimita el objeto (bbox manual o detección automática) y obtiene un recorte RGBA, que luego puede componerse sobre un fondo de estudio procedural o una escena del catálogo local.

El procesamiento pesado (segmentación con FastSAM, composición) vive en el **backend**; el frontend es una SPA que dibuja el bbox y muestra resultados.

## Comandos

### Backend (desde la raíz del proyecto)
```powershell
.\venv\Scripts\activate          # activar el venv (Windows)
pip install -r requirements.txt
cd backend\app
uvicorn main:app --reload        # arranca en http://localhost:8000
```
El servidor **debe arrancarse desde `backend/app`**: los imports son planos (`from segmentation import ...`, `from composition import ...`), no paquetes con prefijo. Al iniciar (`lifespan`) precarga el modelo FastSAM y escanea el catálogo de fondos.

Scripts de smoke/profiling sueltos en `backend/` (`_smoke_studio.py`, `_smoke_cast.py`, `_prof.py`) — útiles para probar composición sin levantar el servidor. No hay suite de tests formal.

### Frontend
```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # build de producción
npm run lint       # ESLint
```

## Requisitos

- Python 3.10+ y Node.js 18+.
- Pesos del modelo en `backend/app/models/FastSAM-s.pt` (no versionados; descargar de Ultralytics FastSAM). Sin ellos el backend no arranca.

## Arquitectura del backend (`backend/app/`)

- **`main.py`** — App FastAPI, CORS (abierto a `*`), parsing de multipart y todos los endpoints. El catálogo de fondos (`BackgroundCatalog`) y el modelo son instancias globales reutilizadas entre peticiones.
- **`segmentation.py`** — `segment_with_fastsam(pil_image, bbox, debug) -> bytes`. Pipeline de 5 etapas: letterbox a 1024px (estilo YOLO, guardando metadata de escala/padding) → inferencia FastSAM → **selección de máscara con 5 estrategias en cascada** (unión de fragmentos en bbox, box prompt nativo, box prompt por IoU, point prompt, score compuesto) → postprocesado revirtiendo padding + clip al bbox + descarte de componentes que tocan el borde → composición RGBA. El modelo se carga lazy vía `_get_model()`.
- **`utils.py`** — `detectar_bbox_canny()` (bbox automático con Canny adaptativo, usado por `/segment_auto/`) y `pre_procesar_objeto_universal()` (limpieza de halos alfa).
- **`composition/`** — pipeline procedural sin dependencias externas. `studio.py` (fondo procedural blanco/gris + sombra) y `scene.py` (fondo del catálogo + armonización) son los puntos de entrada exportados en `__init__.py`. Módulos de apoyo: `catalog.py` (escaneo de `backgrounds/` por categoría + metadatos `ground_y`/`light_dir`/`reflective`), `harmonization.py` (transferencia de color LAB), `placement.py`, `relighting.py`, `reflections.py`, `shadows.py`, `edges.py` (limpieza de bordes alfa), `io_utils.py`.

### Endpoints
`GET /` (health) · `POST /segment_bbox/` (bbox manual) · `POST /segment_auto/` (Canny→FastSAM) · `POST /compose/studio/` · `POST /compose/scene/` · `GET /backgrounds/` · `POST /backgrounds/rescan/` · `GET /backgrounds/full|thumb/{category}/{name}`.

Convención clave: las coordenadas de bbox y posición viajan como **JSON en campos de form-data**, y el bbox usa **coordenadas relativas `[x1,y1,x2,y2]` en `[0,1]`**.

## Arquitectura del frontend (`frontend/src/`)

SPA React 19 + Vite + Tailwind 4 + **react-router-dom v7**. Rutas en `App.jsx` (rutas en español): `/` (Home), `/profesionalizar` (Pipeline — el flujo de segmentación/composición), `/galeria`, `/mision`, `/como-funciona`, `/saber-mas`; catch-all redirige a `/`.

- **`pages/Pipeline/Pipeline.jsx`** — el componente con lógica real: canvas interactivo de bbox y llamadas a la API. El dibujo del bbox compensa el `object-fit: contain` del `<img>` para convertir coordenadas de pantalla ↔ relativas de imagen (`getImageBounds()`), y descarta cajas accidentales (<1% del tamaño).
- Resto de páginas: contenido mayormente estático/marketing. `components/NavBar/` y `components/PlaceholderPage.jsx` son compartidos.

## Notas

- Las carpetas `backend/app/debug_segmentation/` y `debug_auto/` reciben imágenes de debug cuando se pasa `debug=True`.
- El frontend asume el backend en `http://localhost:8000`; CORS está abierto, así que ambos dev servers conviven sin proxy.
- El README.md describe la UI como single-page de dos paneles: está desactualizado respecto al router multi-página actual (la lógica vive en Pipeline).

## Instrucciones

- Tu prioridad es ser correcto, no sonar seguro.
- Si no estás seguro de algo, dilo claramente.
- Usa "No estoy seguro, pero..." o "Deberías verificar esto".
- Nunca afirmes como hecho lo que es una estimación.
- Si falta contexto para responder bien, pídelo.
- No inventes fuentes. Nunca.
- Si no puedes nombrar una fuente verificable, dilo.
- Prioriza documentación oficial y fuentes primarias.
- Si una fuente puede estar desactualizada, avisa.
- Marca cualquier estadística que no puedas confirmar.
- Si no hay dato exacto, di que no lo hay.
- Para eventos recientes, advierte que puede haber cambiado.
- Nunca atribuyas una cita a alguien si no estás seguro.
- Separa siempre hechos confirmados de interpretaciones.