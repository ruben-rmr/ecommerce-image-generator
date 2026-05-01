# Ecommerce Image Generator

Aplicación web para la **segmentación automática de productos** en imágenes de e-commerce y la **generación de fondos personalizados** mediante IA. El usuario sube una fotografía de producto, delimita el objeto con un bounding box interactivo y obtiene un recorte con fondo transparente listo para uso comercial.

---

## Tabla de contenidos

1. [Arquitectura general](#arquitectura-general)
2. [Stack tecnológico](#stack-tecnológico)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Backend](#backend)
   - [Endpoints de la API](#endpoints-de-la-api)
   - [Módulo de segmentación](#módulo-de-segmentación-segmentationpy)
   - [Módulo de generación de fondo](#módulo-de-generación-de-fondo-generationpy)
   - [Utilidades de preprocesamiento](#utilidades-de-preprocesamiento-utilspy)
5. [Frontend](#frontend)
6. [Instalación y ejecución](#instalación-y-ejecución)
7. [Requisitos del sistema](#requisitos-del-sistema)
8. [APIs externas](#apis-externas)

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
                                               │     APIs externas            │
                                               │  · Replicate (SAM2)          │
                                               │  · Photoroom (generación)    │
                                               └──────────────────────────────┘
```

El frontend envía la imagen y las coordenadas del bounding box al backend. El backend delega la segmentación al modelo **SAM2 (Segment Anything Model 2)** de Meta a través de la API de **Replicate**, compone el resultado RGBA y lo devuelve al cliente como PNG con fondo transparente.

---

## Stack tecnológico

### Backend

| Tecnología | Versión / Detalle | Propósito |
|---|---|---|
| Python | 3.x | Lenguaje del servidor |
| FastAPI | — | Framework HTTP asíncrono |
| Uvicorn | — | Servidor ASGI |
| Replicate SDK | — | Llamadas al modelo SAM2 en la nube |
| Pillow (PIL) | — | Manipulación de imágenes, composición RGBA |
| OpenCV (`cv2`) | — | Operaciones morfológicas, inpainting, preprocesamiento |
| NumPy | — | Manipulación de arrays/máscaras |
| httpx | — | Cliente HTTP asíncrono para descarga de máscaras |
| requests | — | Cliente HTTP para Photoroom API |
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
│       ├── main.py              # Servidor FastAPI, endpoints y CORS
│       ├── segmentation.py      # Segmentación con SAM2 vía Replicate API
│       ├── generation.py        # Generación de fondos vía Photoroom API
│       ├── utils.py             # Preprocesamiento de imagen (limpieza, feathering)
│       ├── models/
│       │   └── FastSAM-s.pt     # Pesos del modelo FastSAM (variante small)
│       └── _debug_masks/        # Máscaras de debug generadas durante inferencia
├── frontend/
│   ├── index.html               # Punto de entrada HTML
│   ├── package.json             # Dependencias y scripts npm
│   ├── vite.config.js           # Configuración de Vite
│   ├── tailwind.config.js       # Configuración de Tailwind CSS
│   ├── postcss.config.js        # Configuración de PostCSS
│   ├── eslint.config.js         # Configuración de ESLint
│   └── src/
│       ├── main.jsx             # Bootstrap de React (StrictMode + createRoot)
│       ├── App.jsx              # Componente principal de la aplicación
│       ├── App.css              # Estilos de la aplicación
│       └── index.css            # Reset CSS base
├── requirements.txt             # Dependencias Python
├── .gitignore                   # Exclusiones de Git
└── README.md                    # Este archivo
```

---

## Backend

### Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Health check — devuelve estado del servidor |
| `POST` | `/segment_bbox/` | Segmentación de imagen con bounding box |

#### `POST /segment_bbox/`

Recibe una imagen y un bounding box dibujado por el usuario. Devuelve un PNG con el objeto segmentado y fondo transparente.

**Parámetros (multipart/form-data):**

| Campo | Tipo | Descripción |
|---|---|---|
| `file` | `UploadFile` | Imagen del producto (JPEG, PNG, etc.) |
| `bbox` | `string` (JSON) | Array de 4 coordenadas relativas `[x1, y1, x2, y2]` en rango `[0, 1]` |

**Flujo interno:**

1. Parsea y valida las coordenadas del bounding box.
2. Abre la imagen con Pillow y corrige la orientación EXIF (`ImageOps.exif_transpose`).
3. Convierte las coordenadas relativas (0–1) a píxeles absolutos.
4. Re-codifica la imagen corregida como PNG en bytes.
5. Envía imagen + bbox a SAM2 vía Replicate.
6. Devuelve el PNG resultante con `Content-Type: image/png`.

**Respuesta:** `200 OK` con body binario PNG, o `400`/`500` con detalle del error.

**Configuración CORS:** Permite todos los orígenes (`*`), métodos y cabeceras para facilitar el desarrollo local.

---

### Módulo de segmentación (`segmentation.py`)

Gestiona la comunicación con el modelo **SAM2** (`meta/sam-2`) alojado en Replicate.

**Función principal:** `segment_with_sam_api(image_bytes, bbox) -> bytes`

| Paso | Operación |
|---|---|
| 1 | Codifica la imagen en Base64 y la envuelve como `data:` URI |
| 2 | Envía la imagen y el bbox a `replicate.run()` en un hilo separado (`asyncio.to_thread`) para no bloquear el event loop |
| 3 | Recibe la URL de la máscara generada por SAM2 |
| 4 | Descarga la máscara con `httpx` (cliente asíncrono, timeout 60s) |
| 5 | Abre la imagen original (RGB) y la máscara (escala de grises) con Pillow |
| 6 | Redimensiona la máscara al tamaño de la original si difieren |
| 7 | Compone la imagen RGBA final aplicando la máscara como canal alfa |

---

### Módulo de generación de fondo (`generation.py`)

Integración con la **API de Photoroom** para generar fondos personalizados a partir de un prompt de texto.

**Función:** `generate_background_via_api(image_bytes, prompt) -> bytes`

- Envía la imagen segmentada (PNG con fondo transparente) al endpoint `v2/edit` de Photoroom.
- Aplica un padding de `0.1` para que el objeto tenga margen visual.
- El prompt describe el fondo deseado (p.ej. "fondo minimalista blanco con sombra suave").
- Devuelve los bytes de la imagen con el nuevo fondo generado.

> **Nota:** Este módulo está implementado pero aún no conectado a un endpoint público en `main.py`.

---

### Utilidades de preprocesamiento (`utils.py`)

**Función:** `pre_procesar_objeto_universal(pil_image_original) -> PIL.Image`

Limpieza avanzada de la imagen segmentada antes de enviarla a APIs externas de generación de fondo. Elimina artefactos de borde (halos blancos) que pueden degradar la calidad del resultado final.

**Pipeline:**

| Paso | Técnica | Detalle |
|---|---|---|
| 1 | **Inpainting Telea** | Rellena píxeles semi-transparentes del borde con colores interpolados del objeto, eliminando halos blancos o sucios |
| 2 | **Erosión elíptica** | Contrae el canal alfa 1 px con kernel elíptico (`MORPH_ELLIPSE` 3x3), respetando curvas naturales del objeto |
| 3 | **Micro-feathering** | Aplica `GaussianBlur(radius=0.8)` al alfa para una transición suave y fotográfica entre objeto y fondo |

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

- 5 puntos de muestreo en cruz centrada con margen del 7% para un prompt de punto robusto.
- Compensación automática del aspect ratio entre el elemento DOM y la imagen natural (`getImageBounds()`).
- Conversión bidireccional coordenadas de pantalla ↔ coordenadas relativas de imagen.
- Descarte de cajas accidentales (< 1% del ancho o alto de la imagen).

---

## Instalación y ejecución

### Prerrequisitos

- Python 3.10+
- Node.js 18+
- Cuenta en [Replicate](https://replicate.com/) con token de API configurado (`REPLICATE_API_TOKEN`)

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

El servidor arranca en `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

El cliente arranca en `http://localhost:5173` (puerto por defecto de Vite).

### Variables de entorno

| Variable | Descripción |
|---|---|
| `REPLICATE_API_TOKEN` | Token de autenticación para la API de Replicate |

---

## Requisitos del sistema

| Componente | Mínimo recomendado |
|---|---|
| RAM | 8 GB |
| GPU | Opcional (la inferencia se ejecuta en la nube vía Replicate) |
| Almacenamiento | ~200 MB (dependencias + pesos FastSAM-s.pt) |
| Red | Conexión a Internet requerida para las APIs de Replicate y Photoroom |

---

## APIs externas

### Replicate — SAM2 (Meta)

- **Modelo:** `meta/sam-2`
- **Propósito:** Segmentación semántica de objetos mediante bounding box.
- **Tipo de entrada:** Imagen (Base64 data URI) + coordenadas de bounding box en píxeles.
- **Tipo de salida:** URL a la máscara binaria generada (escala de grises).

### Photoroom — Image Editing API

- **Endpoint:** `https://image-api.photoroom.com/v2/edit`
- **Propósito:** Generación de fondos a partir de prompt textual sobre imagen segmentada.
- **Tipo de entrada:** Imagen PNG (multipart) + prompt de fondo.
- **Tipo de salida:** Imagen con fondo generado.
