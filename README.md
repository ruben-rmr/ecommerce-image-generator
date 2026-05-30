# Ecommerce Image Generator

Aplicación web para la **segmentación automática de productos** en imágenes de e-commerce y la **composición de fondos personalizados** de forma totalmente local, sin dependencias de APIs externas de pago. El usuario sube una fotografía de producto, delimita el objeto con un bounding box interactivo (o usa la detección automática), y obtiene un recorte con fondo transparente listo para composición.

---

## Tabla de contenidos

1. [Arquitectura general](#arquitectura-general)
2. [Stack tecnológico](#stack-tecnológico)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Backend](#backend)
   - [Endpoints de la API](#endpoints-de-la-api)
   - [Módulo de segmentación](#módulo-de-segmentación-segmentationpy)
   - [Pipeline de composición local](#pipeline-de-composición-local-composition)
   - [Utilidades](#utilidades-utilspy)
5. [Frontend](#frontend)
6. [Instalación y ejecución](#instalación-y-ejecución)
7. [Requisitos del sistema](#requisitos-del-sistema)

---

## Arquitectura general

```
┌─────────────────────────┐       HTTP        ┌──────────────────────────────┐
│       Frontend          │  ◄──────────────►  │          Backend             │
│  React 19 + Vite + CSS  │   localhost:5173   │  FastAPI + Uvicorn           │
│                         │        ↔           │  localhost:8000              │
└─────────────────────────┘   fetch / CORS     └──────────┬───────────────────┘
                                                          │
                                               ┌──────────▼───────────────────┐
                                               │     Procesamiento local      │
                                               │  · FastSAM (ultralytics)     │
                                               │  · Composición procedural    │
                                               │  · Catálogo de fondos local  │
                                               └──────────────────────────────┘
```

El frontend envía la imagen y las coordenadas del bounding box al backend. El backend ejecuta **FastSAM** localmente sobre la imagen para generar la máscara, compone el resultado RGBA y lo devuelve como PNG con fondo transparente. La composición de fondos también se realiza en local: fondos procedurales de estudio o imágenes locales con armonización de color.

**No se usa ninguna API externa de terceros.** Todo el procesamiento ocurre en la máquina local.

---

## Stack tecnológico

### Backend

| Tecnología | Versión / Detalle | Propósito |
|---|---|---|
| Python | 3.10+ | Lenguaje del servidor |
| FastAPI | — | Framework HTTP asíncrono |
| Uvicorn | — | Servidor ASGI |
| ultralytics | — | Carga y ejecución de FastSAM local |
| Pillow (PIL) | — | Manipulación de imágenes, composición RGBA |
| OpenCV (`cv2`) | — | Preprocesado, morfología, Canny, inpainting |
| NumPy | — | Manipulación de arrays y máscaras |
| python-dotenv | — | Carga de variables de entorno |

### Frontend

| Tecnología | Versión | Propósito |
|---|---|---|
| React | 19.2 | Librería de UI |
| Vite | 7.2 | Bundler y servidor de desarrollo |
| Tailwind CSS | 4.1 | Framework de utilidades CSS |
| PostCSS + Autoprefixer | — | Procesamiento de estilos |
| ESLint | 9.x | Linting de código |

---

## Estructura del proyecto

```
ecommerce-image-generator/
├── backend/
│   └── app/
│       ├── main.py                  # Servidor FastAPI, endpoints y CORS
│       ├── segmentation.py          # Pipeline FastSAM local (preprocesado, inferencia, postprocesado)
│       ├── utils.py                 # Detección automática de bbox (Canny) y limpieza de halos
│       ├── composition/
│       │   ├── __init__.py          # Exporta compose_studio y compose_scene
│       │   ├── studio.py            # Composición sobre fondo procedural de estudio
│       │   ├── scene.py             # Composición sobre fondo local con armonización
│       │   ├── catalog.py           # Catálogo de fondos locales (scan + metadatos)
│       │   ├── edges.py             # Limpieza de bordes alfa (clean_alpha_edges)
│       │   ├── harmonization.py     # Transferencia de color LAB para armonizar escena
│       │   ├── placement.py         # Posicionado y escalado automático del objeto
│       │   ├── relighting.py        # Ajuste de iluminación global del objeto
│       │   ├── reflections.py       # Generación de reflexiones en superficies
│       │   ├── shadows.py           # Sombras de suelo (cast shadows)
│       │   └── io_utils.py          # Lectura/escritura de imágenes PNG
│       ├── models/
│       │   └── FastSAM-s.pt         # Pesos del modelo FastSAM (variante small)
│       ├── backgrounds/             # Fondos locales organizados por categoría
│       ├── debug_segmentation/      # Imágenes de debug generadas por el pipeline de segmentación
│       └── debug_auto/              # Imágenes de debug generadas por la detección Canny
├── frontend/
│   ├── index.html                   # Punto de entrada HTML
│   ├── package.json                 # Dependencias y scripts npm
│   ├── vite.config.js               # Configuración de Vite
│   ├── tailwind.config.js           # Configuración de Tailwind CSS
│   ├── postcss.config.js            # Configuración de PostCSS
│   ├── eslint.config.js             # Configuración de ESLint
│   └── src/
│       ├── main.jsx                 # Bootstrap de React (StrictMode + createRoot)
│       ├── App.jsx                  # Componente principal de la aplicación
│       ├── App.css                  # Estilos de la aplicación
│       └── index.css                # Reset CSS base
├── requirements.txt                 # Dependencias Python
├── .gitignore                       # Exclusiones de Git
└── README.md                        # Este archivo
```

---

## Backend

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Health check — devuelve estado y modos disponibles |
| `POST` | `/segment_bbox/` | Segmentación con bounding box manual |
| `POST` | `/segment_auto/` | Segmentación automática (Canny → bbox → FastSAM) |
| `POST` | `/compose/studio/` | Composición sobre fondo procedural de estudio |
| `POST` | `/compose/scene/` | Composición sobre fondo local con armonización |
| `GET` | `/backgrounds/` | Lista el catálogo de fondos locales agrupados por categoría |
| `POST` | `/backgrounds/rescan/` | Re-escanea la carpeta de fondos en caliente |
| `GET` | `/backgrounds/full/{category}/{name}` | Sirve un fondo a resolución completa |
| `GET` | `/backgrounds/thumb/{category}/{name}` | Sirve una miniatura del fondo (256×256 JPEG) |

#### `POST /segment_bbox/`

Recibe una imagen y un bounding box dibujado por el usuario. Devuelve un PNG con el objeto segmentado y fondo transparente.

**Parámetros (multipart/form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | Imagen del producto (JPEG, PNG, etc.) |
| `bbox` | `string` (JSON) | Array `[x1, y1, x2, y2]` con coordenadas relativas en `[0, 1]` |
| `max_side` | `int` (opcional) | Limita el lado mayor de la imagen antes de segmentar |

#### `POST /segment_auto/`

Detecta automáticamente el objeto principal mediante Canny edge detection y lanza FastSAM sobre el bbox calculado.

**Parámetros:** `file`, `max_side` (igual que `/segment_bbox/`, sin `bbox`).

#### `POST /compose/studio/`

Compone el PNG segmentado sobre un fondo procedural de estudio (blanco o gris suave) con sombras de suelo.

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | PNG con fondo transparente |
| `style` | `string` | `'white'` \| `'soft_gray'` |
| `position` | `string` (JSON, opcional) | `[rel_cx, rel_y_feet]` para posicionado manual |
| `scale` | `float` (opcional) | Ratio altura silueta / altura canvas |
| `canvas` | `string` (JSON, opcional) | `[W, H]` en píxeles. Por defecto `1024×1024` |

#### `POST /compose/scene/`

Compone el PNG segmentado sobre un fondo local del catálogo con armonización de color.

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | PNG con fondo transparente |
| `background_id` | `string` | `'<categoria>/<nombre>'` del catálogo |
| `position` | `string` (JSON, opcional) | Override de posición |
| `scale` | `float` (opcional) | Override de escala |
| `harmonize_strength` | `float` | Intensidad de armonización LAB (0–1, por defecto `0.45`) |
| `canvas` | `string` (JSON, opcional) | Tamaño del canvas. Por defecto = tamaño del fondo |

---

### Módulo de segmentación (`segmentation.py`)

Ejecuta **FastSAM** localmente sobre la imagen del usuario. El modelo se carga una única vez al arrancar el servidor (`lifespan`) y se reutiliza en todas las peticiones.

**Función principal:** `segment_with_fastsam(pil_image, bbox, debug) -> bytes`

El pipeline completo tiene cinco etapas:

| Etapa | Operación |
|---|---|
| 1 · Preprocesado | Resize manteniendo aspect ratio + padding centrado en canvas cuadrado de 1024 px (mismo letterbox que YOLO). Se almacena el metadata de escala y padding para revertirlo después. |
| 2 · Inferencia | FastSAM se ejecuta sobre el canvas preprocesado. Se aplican dos filtros a las máscaras brutas: descarte de máscaras que tocan los bordes de la imagen y descarte de máscaras cuya área queda mayoritariamente fuera del bbox del usuario. |
| 3 · Selección de máscara | Se intentan cinco estrategias en orden: unión de fragmentos contenidos en el bbox (reconstituye objetos que FastSAM devuelve segmentados), box prompt nativo de FastSAM, box prompt por IoU, point prompt (centro del bbox), y selección por score compuesto (cobertura × overlap × centroide × alineamiento con bordes Canny). |
| 4 · Postprocesado | La máscara se recorta al tamaño original revirtiendo el padding y la escala. Se aplica un clip estricto al bbox del usuario, se eliminan componentes conexos que tocan el borde de la imagen, y se descartan fragmentos pequeños desconectados del objeto principal. |
| 5 · Composición RGBA | La máscara binaria se aplica como canal alfa sobre la imagen original RGB y se serializa como PNG. |

---

### Pipeline de composición local (`composition/`)

Módulo sin dependencias externas que genera imágenes de producto listas para e-commerce a partir de un PNG con fondo transparente.

#### Modo estudio (`studio.py`)

Genera un fondo procedural (blanco `#FFFFFF` o gris suave) con gradiente de viñeta y sombra de suelo. El objeto se posiciona y escala automáticamente para ocupar el 80% de la altura del canvas, centrado horizontalmente y apoyado en el tercio inferior.

#### Modo escena (`scene.py`)

Compone el objeto sobre una imagen de fondo local del catálogo. Incluye:

- **Armonización de color** (`harmonization.py`): transferencia de media/desviación en espacio LAB del fondo al objeto, con intensidad controlada por `harmonize_strength`.
- **Posicionado automático** (`placement.py`): calcula la posición y escala óptimas según metadatos del fondo (`ground_y`, `light_dir`).
- **Sombras** (`shadows.py`): sombras de suelo proyectadas según la dirección de luz del fondo.
- **Reflexiones** (`reflections.py`): para fondos marcados como `reflective` en el catálogo.
- **Relighting** (`relighting.py`): ajuste de la luminancia global del objeto para casar con la exposición del fondo.
- **Limpieza de bordes** (`edges.py`): erosión elíptica + micro-feathering del canal alfa para eliminar halos.

#### Catálogo de fondos (`catalog.py`)

Escanea la carpeta `backgrounds/` al arrancar el servidor. Las imágenes se organizan por subcarpeta (categoría). Los metadatos opcionales (posición del suelo, dirección de luz, superficie reflectante) se pueden definir junto a cada imagen y son leídos por el modo escena.

---

### Utilidades (`utils.py`)

**`detectar_bbox_canny(pil_image, padding) -> [x1, y1, x2, y2]`**

Detecta el bounding box del objeto principal usando Canny edge detection con umbrales adaptativos basados en la mediana de la imagen. Se usa en el endpoint `/segment_auto/`.

**`pre_procesar_objeto_universal(pil_image) -> PIL.Image`**

Limpieza de halos del canal alfa: inpainting Telea en píxeles semi-transparentes → erosión elíptica 1 px → micro-feathering gaussiano (radius 0.8). La pipeline de composición usa una versión equivalente en `composition/edges.py`.

---

## Frontend

Aplicación de página única (SPA) construida con React 19 y Vite.

### Interfaz

La interfaz se divide en dos paneles:

- **Panel izquierdo (entrada):** Zona de carga de imagen (drag & drop o selector de archivos) con canvas interactivo para dibujar el bounding box.
- **Panel derecho (resultado):** Previsualización del resultado segmentado sobre fondo de tablero de ajedrez (indicador de transparencia), con botón de descarga.

### Flujo de interacción

1. **Carga de imagen:** El usuario arrastra o selecciona un archivo de imagen.
2. **Dibujo del bounding box:** Click + drag sobre la imagen para delimitar el objeto. Las coordenadas se calculan en posición relativa (0–1) respecto a la imagen real, compensando el `object-fit: contain` del elemento `<img>`.
3. **Segmentación:** El botón "Segmentar" envía la imagen y el bbox al endpoint `/segment_bbox/`.
4. **Resultado:** El PNG con fondo transparente se muestra en el panel derecho y se puede descargar.

### Detalles técnicos del canvas de bounding box

- Compensación automática del aspect ratio entre el elemento DOM y la imagen natural (`getImageBounds()`).
- Conversión bidireccional coordenadas de pantalla ↔ coordenadas relativas de imagen.
- Descarte de cajas accidentales (< 1% del ancho o alto de la imagen).

---

## Instalación y ejecución

### Prerrequisitos

- Python 3.10+
- Node.js 18+
- Pesos del modelo en `backend/app/models/FastSAM-s.pt` (descargables desde [Ultralytics FastSAM](https://github.com/ultralytics/assets/releases))

**No se necesita ninguna clave de API externa.**

### Backend

```bash
# Desde la raíz del proyecto
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt

# Iniciar el servidor
cd backend/app
uvicorn main:app --reload
```

El servidor arranca en `http://localhost:8000`. Al iniciarse, carga el modelo FastSAM en memoria y escanea el catálogo de fondos.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

El cliente arranca en `http://localhost:5173` (puerto por defecto de Vite).

---

## Requisitos del sistema

| Componente | Mínimo recomendado |
|---|---|
| RAM | 8 GB |
| GPU | Opcional (FastSAM se ejecuta en CPU por defecto) |
| Almacenamiento | ~200 MB (dependencias + pesos `FastSAM-s.pt`) |
| Red | No requerida (procesamiento 100% local) |
