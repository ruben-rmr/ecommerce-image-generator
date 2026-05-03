import { useState, useRef, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [imageFile, setImageFile]     = useState(null)
  const [imageObj, setImageObj]       = useState(null)
  const [bbox, setBbox]               = useState(null)
  const [drawing, setDrawing]         = useState(null)
  const [segResult, setSegResult]     = useState(null)   // blob URL del PNG segmentado
  const [segBlob, setSegBlob]         = useState(null)    // blob raw para enviar a /generate
  const [genResult, setGenResult]     = useState(null)    // blob URL de la imagen generada
  const [loading, setLoading]         = useState(false)
  const [generating, setGenerating]   = useState(false)
  const [error, setError]             = useState(null)
  const [fileDragging, setFileDragging] = useState(false)

  // Prompts predefinidos
  const [prompts, setPrompts]         = useState({})
  const [selectedPrompt, setSelectedPrompt] = useState('estudio_blanco')

  const imgRef    = useRef(null)
  const fileInput = useRef(null)

  // Cargar prompts disponibles al montar
  useEffect(() => {
    fetch(`${API_URL}/prompts/`)
      .then(r => r.json())
      .then(data => {
        setPrompts(data)
        const keys = Object.keys(data)
        if (keys.length > 0) setSelectedPrompt(keys[0])
      })
      .catch(() => {})
  }, [])

  // ── Utilidad: obtener área real pintada de la imagen dentro del <img> ──
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

  const screenToRel = useCallback((clientX, clientY) => {
    const b = getImageBounds()
    if (!b) return null
    const rx = Math.max(0, Math.min(1, (clientX - b.left) / b.width))
    const ry = Math.max(0, Math.min(1, (clientY - b.top)  / b.height))
    return { rx, ry }
  }, [getImageBounds])

  // ── Carga de imagen ──────────────────────────────────────────────────
  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImageFile(file)
    setBbox(null)
    setDrawing(null)
    setSegResult(null)
    setSegBlob(null)
    setGenResult(null)
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

  // ── Dibujo del bounding box ──────────────────────────────────────────
  const onMouseDown = (e) => {
    if (!imageObj) return
    e.preventDefault()
    const rel = screenToRel(e.clientX, e.clientY)
    if (!rel) return
    setDrawing({ x1: rel.rx, y1: rel.ry, x2: rel.rx, y2: rel.ry })
    setBbox(null)
    setSegResult(null)
    setSegBlob(null)
    setGenResult(null)
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

    if ((final.x2 - final.x1) > 0.01 && (final.y2 - final.y1) > 0.01) {
      setBbox(final)
    }
    setDrawing(null)
  }

  const getRectStyle = (box) => {
    if (!box || !imgRef.current || !imageObj) return null
    const bounds = getImageBounds()
    if (!bounds) return null

    const wrapperRect = imgRef.current.parentElement.getBoundingClientRect()

    const left   = (bounds.left - wrapperRect.left) + Math.min(box.x1, box.x2) * bounds.width
    const top    = (bounds.top  - wrapperRect.top)  + Math.min(box.y1, box.y2) * bounds.height
    const width  = Math.abs(box.x2 - box.x1) * bounds.width
    const height = Math.abs(box.y2 - box.y1) * bounds.height

    return { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }
  }

  // ── Segmentar (manual con bbox) ──────────────────────────────────────
  const handleSegment = async () => {
    if (!imageFile || !bbox) return
    setLoading(true)
    setError(null)
    setSegResult(null)
    setSegBlob(null)
    setGenResult(null)

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
      setSegBlob(blob)
      setSegResult(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  // ── Segmentación automática ──────────────────────────────────────────
  const handleAutoSegment = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    setSegResult(null)
    setSegBlob(null)
    setGenResult(null)

    try {
      const fd = new FormData()
      fd.append('file', imageFile)

      const res = await fetch(`${API_URL}/segment_auto/`, { method: 'POST', body: fd })

      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }

      const blob = await res.blob()
      setSegBlob(blob)
      setSegResult(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  // ── Generar fondo con Photoroom ──────────────────────────────────────
  const handleGenerate = async () => {
    if (!segBlob) return
    setGenerating(true)
    setError(null)

    try {
      const fd = new FormData()
      fd.append('file', segBlob, 'segmentado.png')
      fd.append('prompt_key', selectedPrompt)

      const res = await fetch(`${API_URL}/generate/`, { method: 'POST', body: fd })

      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }

      const blob = await res.blob()
      setGenResult(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setGenerating(false)
    }
  }

  const canSegment     = imageFile && imageObj && bbox && !loading && !generating
  const canAutoSegment = imageFile && imageObj && !loading && !generating
  const canGenerate    = segBlob && !loading && !generating
  const activeBox      = drawing || bbox

  // Qué mostrar en el panel derecho
  const rightImage = genResult || segResult
  const rightTitle = genResult ? 'Imagen generada' : 'Segmentación'
  const rightAlt   = genResult ? 'Imagen generada' : 'Segmentación'
  const isProcessing = loading || generating

  const PROMPT_LABELS = {
    estudio_blanco: 'Estudio blanco profesional',
  }

  return (
    <div className="app">
      <header className="header">
        <h1>FastSAM · Segmentación y generación de fondos</h1>
        <p>Sube una imagen, segmenta el objeto y genera un fondo profesional con IA.</p>
      </header>

      <main className="main">
        {/* ── Panel izquierdo: imagen original ─────────────────────────── */}
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
              <button className="btn-secondary" onClick={() => {
                setImageObj(null); setImageFile(null); setBbox(null); setDrawing(null)
                setSegResult(null); setSegBlob(null); setGenResult(null); setError(null)
              }}>
                Nueva imagen
              </button>
              <button className="btn-primary" onClick={handleAutoSegment} disabled={!canAutoSegment}>
                {loading && !bbox ? <span className="spinner" /> : 'Auto Segmentar'}
              </button>
              <button className="btn-primary" onClick={handleSegment} disabled={!canSegment}>
                {loading && bbox ? <span className="spinner" /> : 'Segmentar'}
              </button>
            </div>
          )}

          {/* Selector de prompt + botón generar */}
          {segResult && !generating && (
            <div className="generate-section">
              <label className="prompt-label" htmlFor="prompt-select">Fondo:</label>
              <select
                id="prompt-select"
                className="prompt-select"
                value={selectedPrompt}
                onChange={(e) => setSelectedPrompt(e.target.value)}
              >
                {Object.keys(prompts).map(key => (
                  <option key={key} value={key}>
                    {PROMPT_LABELS[key] || key}
                  </option>
                ))}
              </select>
              <button className="btn-generate" onClick={handleGenerate} disabled={!canGenerate}>
                Generar fondo
              </button>
            </div>
          )}

          {!bbox && !drawing && imageObj && !loading && !segResult && (
            <p className="hint">Pulsa <strong>Auto Segmentar</strong> o dibuja un recuadro manualmente sobre el objeto.</p>
          )}
          {bbox && !segResult && !loading && (
            <p className="hint">Recuadro listo — pulsa <strong>Segmentar</strong>. Puedes redibujar o usar <strong>Auto Segmentar</strong>.</p>
          )}
          {segResult && !genResult && !generating && (
            <p className="hint">Segmentación lista. Selecciona un fondo y pulsa <strong>Generar fondo</strong>.</p>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        {/* ── Panel derecho: resultado ─────────────────────────────────── */}
        <section className={`panel result-panel ${rightImage ? 'visible' : ''}`}>
          {isProcessing && (
            <div className="loading-state">
              <span className="spinner large" />
              <p>{generating ? 'Generando fondo…' : 'Segmentando…'}</p>
            </div>
          )}
          {rightImage && !isProcessing && (
            <>
              <h2 className="result-title">{rightTitle}</h2>
              <div className="result-img-wrapper">
                <img src={rightImage} alt={rightAlt} className="result-img" />
              </div>
              <a
                className="btn-download"
                href={rightImage}
                download={genResult ? 'generado.png' : 'segmentado.png'}
              >
                Descargar PNG
              </a>
            </>
          )}
          {!rightImage && !isProcessing && (
            <div className="result-placeholder">
              <span>El resultado aparecerá aquí</span>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
