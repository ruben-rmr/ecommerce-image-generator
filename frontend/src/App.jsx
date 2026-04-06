import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [imageFile, setImageFile]     = useState(null)
  const [imageObj, setImageObj]       = useState(null)   // Image() para dimensiones naturales
  const [bbox, setBbox]               = useState(null)   // { x1, y1, x2, y2 } relativas 0-1 (finalizado)
  const [drawing, setDrawing]         = useState(null)   // { x1, y1, x2, y2 } mientras arrastra
  const [result, setResult]           = useState(null)   // blob URL del PNG resultante
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [fileDragging, setFileDragging] = useState(false)

  const imgRef    = useRef(null)
  const fileInput = useRef(null)

  // ── Utilidad: obtener área real pintada de la imagen dentro del <img> ──────
  // Con object-fit:contain, la imagen puede tener barras laterales/superiores.
  const getImageBounds = useCallback(() => {
    if (!imgRef.current || !imageObj) return null
    const rect      = imgRef.current.getBoundingClientRect()
    const elAspect  = rect.width / rect.height
    const imgAspect = imageObj.width / imageObj.height

    let imgLeft, imgTop, imgW, imgH
    if (imgAspect > elAspect) {
      imgW    = rect.width
      imgH    = rect.width / imgAspect
      imgLeft = rect.left
      imgTop  = rect.top + (rect.height - imgH) / 2
    } else {
      imgH    = rect.height
      imgW    = rect.height * imgAspect
      imgTop  = rect.top
      imgLeft = rect.left + (rect.width - imgW) / 2
    }
    return { left: imgLeft, top: imgTop, width: imgW, height: imgH }
  }, [imageObj])

  // Convertir coordenadas de pantalla a relativas 0-1 de la imagen
  const screenToRel = useCallback((clientX, clientY) => {
    const b = getImageBounds()
    if (!b) return null
    const rx = Math.max(0, Math.min(1, (clientX - b.left) / b.width))
    const ry = Math.max(0, Math.min(1, (clientY - b.top)  / b.height))
    return { rx, ry }
  }, [getImageBounds])

  // ── Carga de imagen ───────────────────────────────────────────────────────
  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImageFile(file)
    setBbox(null)
    setDrawing(null)
    setResult(null)
    setError(null)

    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => setImageObj({ url, width: img.naturalWidth, height: img.naturalHeight })
    img.src = url
  }, [])

  const onFileChange = (e) => loadFile(e.target.files[0])
  const onDrop = (e) => {
    e.preventDefault()
    setFileDragging(false)
    loadFile(e.dataTransfer.files[0])
  }

  // ── Dibujo del bounding box con click + drag ──────────────────────────────
  const onMouseDown = (e) => {
    if (!imageObj) return
    e.preventDefault()
    const rel = screenToRel(e.clientX, e.clientY)
    if (!rel) return
    setDrawing({ x1: rel.rx, y1: rel.ry, x2: rel.rx, y2: rel.ry })
    setBbox(null)
    setResult(null)
    setError(null)
  }

  const onMouseMove = (e) => {
    if (!drawing) return
    const rel = screenToRel(e.clientX, e.clientY)
    if (!rel) return
    setDrawing(prev => prev ? { ...prev, x2: rel.rx, y2: rel.ry } : null)
  }

  const onMouseUp = (e) => {
    if (!drawing) return
    const rel = screenToRel(e.clientX, e.clientY)
    if (!rel) return

    const final = {
      x1: Math.min(drawing.x1, rel.rx),
      y1: Math.min(drawing.y1, rel.ry),
      x2: Math.max(drawing.x1, rel.rx),
      y2: Math.max(drawing.y1, rel.ry),
    }

    // Descartar cajas demasiado pequeñas (clics accidentales)
    if ((final.x2 - final.x1) > 0.01 && (final.y2 - final.y1) > 0.01) {
      setBbox(final)
    }
    setDrawing(null)
  }

  // ── Calcular estilo CSS del rectángulo sobre la imagen ────────────────────
  const getRectStyle = (box) => {
    if (!box || !imgRef.current || !imageObj) return null
    const bounds = getImageBounds()
    if (!bounds) return null

    // Coordenadas relativas al viewport → relativas al image-wrapper (position:relative)
    const wrapperRect = imgRef.current.parentElement.getBoundingClientRect()

    const left   = (bounds.left - wrapperRect.left) + Math.min(box.x1, box.x2) * bounds.width
    const top    = (bounds.top  - wrapperRect.top)  + Math.min(box.y1, box.y2) * bounds.height
    const width  = Math.abs(box.x2 - box.x1) * bounds.width
    const height = Math.abs(box.y2 - box.y1) * bounds.height

    return { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }
  }

  // ── Enviar al backend ─────────────────────────────────────────────────────
  const handleSegment = async () => {
    if (!imageFile || !bbox) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const fd = new FormData()
      fd.append('file', imageFile)
      fd.append('bbox', JSON.stringify([bbox.x1, bbox.y1, bbox.x2, bbox.y2]))

      const res = await fetch(`${API_URL}/segment_bbox/`, { method: 'POST', body: fd })

      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }

      const blob = await res.blob()
      setResult(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  const canSegment = imageFile && imageObj && bbox && !loading
  const activeBox  = drawing || bbox    // caja a dibujar (arrastrando o confirmada)

  return (
    <div className="app">
      <header className="header">
        <h1>FastSAM · Segmentación interactiva</h1>
        <p>Sube una imagen y dibuja un recuadro sobre el objeto para segmentarlo.</p>
      </header>

      <main className="main">
        {/* ── Panel izquierdo: input ────────────────────────────────────── */}
        <section className="panel">
          <div
            className={`drop-zone ${fileDragging ? 'dragging' : ''} ${imageObj ? 'has-image' : ''}`}
            onClick={() => !imageObj && fileInput.current.click()}
            onDragOver={(e) => { e.preventDefault(); setFileDragging(true) }}
            onDragLeave={() => setFileDragging(false)}
            onDrop={onDrop}
          >
            {imageObj ? (
              <div
                className="image-wrapper"
                onMouseDown={onMouseDown}
                onMouseMove={onMouseMove}
                onMouseUp={onMouseUp}
                onMouseLeave={() => drawing && setDrawing(null)}
              >
                <img
                  ref={imgRef}
                  src={imageObj.url}
                  alt="Imagen subida"
                  className="preview-img"
                  draggable={false}
                />
                {activeBox && (() => {
                  const style = getRectStyle(activeBox)
                  return style ? (
                    <div
                      className={`bbox-overlay ${drawing ? 'drawing' : 'done'}`}
                      style={style}
                    />
                  ) : null
                })()}
              </div>
            ) : (
              <div className="drop-hint">
                <span className="drop-icon">📂</span>
                <p>Arrastra una imagen aquí<br/>o <u>haz clic para seleccionar</u></p>
              </div>
            )}
          </div>

          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            hidden
            onChange={onFileChange}
          />

          {imageObj && (
            <div className="actions">
              <button className="btn-secondary" onClick={() => { setImageObj(null); setImageFile(null); setBbox(null); setDrawing(null); setResult(null); setError(null) }}>
                Nueva imagen
              </button>
              <button className="btn-primary" onClick={handleSegment} disabled={!canSegment}>
                {loading ? <span className="spinner" /> : '✂️ Segmentar'}
              </button>
            </div>
          )}

          {!bbox && !drawing && imageObj && !loading && !result && (
            <p className="hint">📦 Haz clic y arrastra sobre el objeto para dibujar un recuadro.</p>
          )}
          {bbox && !result && !loading && (
            <p className="hint">✅ Recuadro listo — pulsa <strong>Segmentar</strong>. Puedes redibujar si lo deseas.</p>
          )}
          {error && <p className="error">⚠️ {error}</p>}
        </section>

        {/* ── Panel derecho: resultado ──────────────────────────────────── */}
        <section className={`panel result-panel ${result ? 'visible' : ''}`}>
          {loading && (
            <div className="loading-state">
              <span className="spinner large" />
              <p>Procesando…</p>
            </div>
          )}
          {result && !loading && (
            <>
              <h2 className="result-title">Resultado</h2>
              <div className="result-img-wrapper">
                <img src={result} alt="Segmentación" className="result-img" />
              </div>
              <a className="btn-download" href={result} download="segmentado.png">
                ⬇️ Descargar PNG
              </a>
            </>
          )}
          {!result && !loading && (
            <div className="result-placeholder">
              <span>El resultado aparecerá aquí</span>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
