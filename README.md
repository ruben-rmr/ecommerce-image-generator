# Documentación técnica — `segmentation.py`

> **Ubicación:** `backend/app/segmentation.py`  
> **Propósito:** Eliminación de fondo de imágenes de producto mediante segmentación semántica con el modelo **FastSAM**.

---

## Índice

1. [Visión general](#visión-general)
2. [Dependencias](#dependencias)
3. [Modelo de IA utilizado](#modelo-de-ia-utilizado)
4. [Flujo de ejecución completo](#flujo-de-ejecución-completo)
5. [Referencia de funciones](#referencia-de-funciones)
   - [`load_model()`](#load_model)
   - [`refine_mask(mask)`](#refine_maskmask)
   - [`select_main_mask(masks_bin, h, w)`](#select_main_maskmasks_bin-h-w)
   - [`bbox_of(binary)`](#bbox_ofbinary)
   - [`bbox_iou(a, b)`](#bbox_ioua-b)
   - [`fuse_fragments(seed_mask, masks_bin, iou_threshold)`](#fuse_fragmentsseed_mask-masks_bin-iou_threshold)
   - [`remove_background(image_cv2)`](#remove_backgroundimage_cv2)
6. [Parámetros de inferencia](#parámetros-de-inferencia)
7. [Decisiones de diseño](#decisiones-de-diseño)
8. [Salida de debug](#salida-de-debug)

---

## Visión general

El módulo tiene un único punto de entrada público: `remove_background()`. Dado que esta función recibe una imagen en formato OpenCV (array NumPy BGR), devuelve la misma imagen con el fondo eliminado en formato **RGBA** (4 canales), utilizando el canal alfa para la transparencia.

El proceso es completamente automático y funciona sin ninguna interacción del usuario ni coordenadas manuales. Internamente usa FastSAM para segmentar todos los objetos de la imagen y después aplica una estrategia de selección + fusión para obtener la máscara del objeto principal.

---

## Dependencias

| Librería | Uso |
|---|---|
| `ultralytics` | Carga y ejecución del modelo FastSAM |
| `torch` | Detección de GPU (CUDA) y gestión de tensores |
| `cv2` (OpenCV) | Operaciones morfológicas, redimensionado, composición RGBA |
| `numpy` | Manipulación de máscaras como arrays |
| `os` | Resolución de rutas absolutas de forma portable |

---

## Modelo de IA utilizado

**FastSAM** (_Fast Segment Anything Model_) es una versión acelerada de SAM (Meta AI) basada en YOLOv8-seg. Opera en dos fases:

1. **Everything mode:** segmenta todos los objetos visibles de la imagen y genera `N` máscaras candidatas, una por región detectada.
2. **Prompt mode:** dada una referencia (punto, caja, texto), selecciona o refina la máscara correspondiente.

El módulo usa el modo _everything_ y luego aplica selección por punto central de forma manual, lo que es equivalente al prompt de punto nativo pero sin la dependencia de `FastSAMPrompt`.

Los pesos del modelo se cargan desde:

```
backend/app/models/FastSAM-s.pt
```

La variante `-s` (small) es más rápida y suficiente para imágenes de producto. Existe también `-x` (extra-large) para mayor precisión si el hardware lo permite.

---

## Flujo de ejecución completo

```
Imagen BGR (OpenCV)
        │
        ▼
┌─────────────────────────────┐
│  FastSAM "everything mode"  │  → N máscaras candidatas (H_mask × W_mask)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Redimensionar todas las    │  → N máscaras binarias (H_orig × W_orig)
│  máscaras al tamaño orig.   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  select_main_mask()         │  → máscara semilla del objeto central
│  (point prompt robusto)     │
└─────────────────────────────┘
        │
        ├── semilla ≥ 8% imagen ──────────────────────────────┐
        │                                                      │
        ▼                                                      │
┌─────────────────────────────┐                               │
│  fuse_fragments()           │  → objeto multicolor: une     │
│  (bbox-IoU iterativo)       │     fragmentos al objeto      │
└─────────────────────────────┘                               │
        │                                                      │
        └──────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  refine_mask()              │  → Close + Open + GaussianBlur
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Composición RGBA           │  → imagen con canal alfa
└─────────────────────────────┘
        │
        ▼
  Imagen RGBA (resultado)
```

---

## Referencia de funciones

### `load_model()`

```python
def load_model() -> None
```

Carga el modelo FastSAM desde disco y lo almacena en la variable global `model`. Se llama **una sola vez** al arrancar el servidor para evitar la costosa carga en cada petición.

**Comportamiento:**
- Construye la ruta al archivo de pesos usando `os.path.abspath(__file__)`, lo que garantiza que funciona independientemente del directorio de trabajo actual.
- Si hay GPU disponible (`torch.cuda.is_available()`), mueve el modelo a CUDA y ejecuta un **warmup** con una imagen negra de 640×640. El warmup precalienta los kernels CUDA y evita que la primera inferencia real sea lenta.
- Si no hay GPU, el modelo corre en CPU con un aviso.

---

### `refine_mask(mask)`

```python
def refine_mask(mask: np.ndarray) -> np.ndarray
```

Limpia y suaviza una máscara binaria para mejorar la calidad visual del recorte. Acepta máscaras con valores en rango `[0, 1]` o `[0, 255]` y siempre devuelve `uint8`.

**Pipeline interno:**

| Paso | Operación | Kernel | Efecto |
|---|---|---|---|
| 1 | `MORPH_CLOSE` | 7×7 | Rellena huecos internos pequeños (p.ej. letras blancas sobre fondo del objeto) |
| 2 | `MORPH_OPEN` | 3×3 | Elimina píxeles de ruido aislados en el exterior de la máscara |
| 3 | `GaussianBlur` | 5×5 | Desenfoca suavemente el borde → transición natural (feathering) |

> **Por qué CLOSE antes que OPEN:** El cierre morfológico rellena zonas interiores sin expandir el contorno exterior. Si se aplicase OPEN primero, se eliminarían fragmentos antes de poder unirlos, perdiendo partes del objeto.

---

### `select_main_mask(masks_bin, h, w)`

```python
def select_main_mask(masks_bin: list[np.ndarray], h: int, w: int) -> tuple[np.ndarray, int]
```

Implementa un **point prompt robusto**: selecciona la máscara más grande que contenga el objeto en el centro de la imagen, sin depender de un único píxel.

**Puntos de muestreo** (5 puntos formando una cruz centrada con margen del 7 %):
```
         ·          ← (cx, cy - 7%)
  ·      ●      ·   ← (cx ± 7%, cy) y centro exacto (cx, cy)
         ·          ← (cx, cy + 7%)
```

**Lógica de selección:**
1. Para cada máscara, comprueba si alguno de los 5 puntos está activo (`== 1`) en ella.
2. De todas las máscaras con hit, se queda con la de **mayor área en píxeles** (descarta ruido pequeño o logos menores).
3. Si ninguna máscara tiene hit en los 5 puntos (objeto muy descentrado), devuelve la máscara globalmente más grande como fallback.

**Devuelve:** `(máscara_binaria, área_en_píxeles)`

---

### `bbox_of(binary)`

```python
def bbox_of(binary: np.ndarray) -> tuple[int,int,int,int] | None
```

Calcula el **bounding box** mínimo de una máscara binaria.

- Proyecta la máscara sobre filas y columnas con `np.any()`.
- Devuelve `(x1, y1, x2, y2)` en coordenadas de imagen.
- Devuelve `None` si la máscara está vacía.

---

### `bbox_iou(a, b)`

```python
def bbox_iou(a: tuple, b: tuple) -> float
```

Calcula el **Intersection over Union (IoU)** entre dos bounding boxes en formato `(x1, y1, x2, y2)`.

```
         IoU = Área(intersección) / Área(unión)
```

Rango: `[0.0, 1.0]`. Devuelve `0.0` si no hay intersección.

---

### `fuse_fragments(seed_mask, masks_bin, iou_threshold)`

```python
def fuse_fragments(
    seed_mask:      np.ndarray,
    masks_bin:      list[np.ndarray],
    iou_threshold:  float = 0.15
) -> np.ndarray
```

Fusiona fragmentos del objeto principal cuando FastSAM lo ha segmentado en varias partes pequeñas (típico en objetos con múltiples colores, diseños gráficos, o etiquetas).

**Algoritmo (expansión iterativa por bounding-box):**

1. Comienza con la `seed_mask` como unión inicial.
2. En cada pasada, calcula el bounding box de la unión actual.
3. Para cada máscara candidata:
   - Si ya está incluida en la unión (>90 % de sus píxeles coinciden), se salta.
   - Si su bounding box tiene IoU ≥ `iou_threshold` con el de la unión, se añade (`np.maximum`).
4. Repite hasta que ninguna máscara nueva pueda añadirse o hasta un máximo de 5 pasadas.

**Por qué bbox-IoU y no solapamiento de píxeles:**  
El solapamiento de píxeles requiere que las máscaras se toquen físicamente. El bbox-IoU es más tolerante: si el bounding box de un fragmento está dentro del área del objeto, se une aunque haya un pequeño hueco entre ellos (p.ej. el espacio entre el tapón y el cuerpo de un envase).

**Por qué es seguro para objetos monocolor:**  
Esta función solo se llama si la semilla cubre menos del 8 % de la imagen. Para un objeto monocolor bien segmentado la semilla siempre supera ese umbral → la función nunca se ejecuta.

---

### `remove_background(image_cv2)`

```python
def remove_background(image_cv2: np.ndarray) -> np.ndarray | None
```

**Función principal y único punto de entrada público del módulo.**

Recibe una imagen en formato OpenCV (BGR, `uint8`) y devuelve la misma imagen con fondo transparente (RGBA, `uint8`), o `None` si no se detecta ningún objeto.

**Parámetros de inferencia usados:**

| Parámetro | Valor | Razón |
|---|---|---|
| `conf` | `0.4` | Captura partes del objeto sin añadir demasiado ruido de fondo |
| `iou` | `0.9` | Suprime máscaras duplicadas agresivamente |
| `imgsz` | `640` | Resolución probada y estable; valores mayores distorsionaban la generación de máscaras |
| `retina_masks` | `True` | Devuelve máscaras de alta resolución en vez de las de baja resolución internas |

**Pasos internos:**

1. **Inferencia** — FastSAM genera `N` máscaras.
2. **Normalización** — todas las máscaras se redimensionan al tamaño original y se binariza con umbral 0.5.
3. **Selección** — `select_main_mask()` identifica la máscara del objeto central.
4. **Fusión condicional** — si la semilla < 8 % de la imagen, `fuse_fragments()` une los fragmentos.
5. **Refinamiento** — `refine_mask()` aplica operaciones morfológicas y suavizado.
6. **Composición** — se separan los canales BGR, se añade la máscara como canal alfa y se combina en RGBA.

---

## Parámetros de inferencia

Los parámetros de FastSAM tienen un impacto decisivo en la calidad de la segmentación:

| Parámetro | Rango típico | Efecto al aumentar |
|---|---|---|
| `conf` | 0.1 – 0.9 | Más máscaras pero más ruido |
| `iou` | 0.1 – 0.95 | Menos duplicados (valor alto = más agresivo) |
| `imgsz` | 320 – 1280 | Más detalle pero puede cambiar el comportamiento de detección |

> ⚠️ **Importante:** Cambiar `imgsz` a valores por encima de 640 puede hacer que FastSAM genere más máscaras pequeñas en objetos simples, lo que rompe la selección por punto central. Modificar este valor requiere validación cuidadosa.

---

## Decisiones de diseño

### ¿Por qué no usar `FastSAMPrompt` directamente?

La clase `FastSAMPrompt` de Ultralytics ofrece un `point_prompt` nativo, pero su API ha cambiado entre versiones (el parámetro `points` puede aceptar distintos formatos según la versión instalada). El enfoque manual de `select_main_mask()` es equivalente en resultado y más robusto ante actualizaciones de la librería.

### ¿Por qué bounding-box IoU para la fusión y no solapamiento de píxeles?

El solapamiento de píxeles requiere que los fragmentos se toquen físicamente. En un envase de zumo, por ejemplo, el área del tapón puede estar separada del cuerpo por un pequeño hueco. El bbox-IoU une correctamente estas partes al comparar sus envolventes rectangulares.

### ¿Por qué el umbral del 8 % para activar la fusión?

En imágenes de producto estándar, el objeto ocupa al menos el 15–40 % del encuadre. Si la máscara principal es menor al 8 %, indica claramente que el objeto fue fragmentado por FastSAM en partes pequeñas. Este umbral evita activar la fusión en objetos monocolor correctamente segmentados.

---

## Salida de debug

Cada llamada a `remove_background()` guarda automáticamente el resultado en:

```
debug_recorte_perfecto.png
```

Este archivo se escribe en el **directorio de trabajo actual** del proceso (normalmente la raíz del backend). Es útil para inspeccionar visualmente la máscara generada sin necesidad de conectarse a la interfaz. Para desactivarlo en producción, basta con comentar la línea `cv2.imwrite(...)`.
