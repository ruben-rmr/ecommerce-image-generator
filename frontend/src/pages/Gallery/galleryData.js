// ── Datos de la galería ───────────────────────────────────────────────
// Las imágenes se descubren AUTOMÁTICAMENTE: basta con dejar los archivos en
// la carpeta ./images/ (junto a este fichero) y aparecerán en la galería.
//
// Título y descripción:
//   1. Por defecto el título se deriva del nombre del archivo
//      ("fondo-marmol.jpg"  ->  "Fondo Marmol").
//   2. Si quieres personalizar título/descripción de una imagen, añádela al
//      objeto META de abajo usando su nombre de archivo como clave.

const modules = import.meta.glob(
  './images/*.{jpg,jpeg,png,webp,avif,gif,JPG,JPEG,PNG,WEBP,AVIF,GIF}',
  { eager: true, import: 'default' }
)

// Metadatos opcionales por nombre de archivo. Ejemplo:
// 'nature.jpg': { title: 'Nature', description: 'Photography' },
const META = {
}

const DEFAULT_DESCRIPTION = 'Estilo de fondo'

function fileNameOf(path) {
  return path.split('/').pop()
}

function prettifyTitle(fileName) {
  return fileName
    .replace(/\.[^.]+$/, '')        // quita la extensión
    .replace(/[-_]+/g, ' ')          // guiones/guiones bajos -> espacios
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export const GALLERY_ITEMS = Object.entries(modules)
  .sort(([a], [b]) => a.localeCompare(b, 'es', { numeric: true }))
  .map(([path, src]) => {
    const fileName = fileNameOf(path)
    const meta = META[fileName] || {}
    return {
      src,
      title: meta.title || prettifyTitle(fileName),
      description: meta.description || DEFAULT_DESCRIPTION,
    }
  })
