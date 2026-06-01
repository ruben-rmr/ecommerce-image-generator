import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import './Pipeline.css'

const API_URL = 'http://localhost:8000'

const STUDIO_STYLES = [
  { key: 'white', label: 'Blanco profesional', desc: 'Fondo blanco con vignette muy suave.' },
]

const RESIZE_OPTIONS = [3840, 2560, 1920, 1280, 960, 640]

function pickClosestResize(maxSide) {
  return RESIZE_OPTIONS.reduce((best, opt) =>
    Math.abs(opt - maxSide) < Math.abs(best - maxSide) ? opt : best
  )
}

// Imágenes que se muestran en el visor de instrucciones (rellena los archivos
// en frontend/public/instructions/ cuando los tengas).
const INSTRUCTION_IMAGES = [
  '/instructions/step-1.png',
  '/instructions/step-2.png',
]

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function shortType(mime) {
  if (!mime) return '—'
  return mime.replace('image/', '').toUpperCase()
}

// mode: 'full' | 'segment' | 'generate'
export default function Pipeline({ mode = 'full' }) {
  const navigate = useNavigate()
  const onBackHome = () => navigate('/')

  const [imageFile, setImageFile] = useState(null)
  const [imageObj, setImageObj] = useState(null)
  const [bbox, setBbox] = useState(null)
  const [drawing, setDrawing] = useState(null)
  const [segResult, setSegResult] = useState(null)
  const [segBlob, setSegBlob] = useState(null)
  const [genResult, setGenResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState(null)
  const [fileDragging, setFileDragging] = useState(false)

  // MODO de composición ('studio' = Estudio profesional, 'scene' = Escena comercial).
  const [composeMode, setComposeMode] = useState('studio')
  const [studioStyle, setStudioStyle] = useState('white')
  const [backgrounds, setBackgrounds] = useState({})
  const [selectedBg, setSelectedBg] = useState(null)
  const [placement, setPlacement] = useState({ x: 0.5, y: 0.78, scale: 0.4 })
  const [harmonize, setHarmonize] = useState(0.45)

  // Resolución máxima antes de pasar al modelo (null = original)
  const [resizeMaxSide, setResizeMaxSide] = useState(640)

  // Tabs principales del panel inferior
  const initialTab = mode === 'generate' ? 'generate' : 'segment'
  const [activeTab, setActiveTab] = useState(initialTab)
  // Sub-pestañas del modo segment
  const [segMode, setSegMode] = useState('auto') // 'auto' | 'manual'
  // Estado del panel inferior
  const [panelCollapsed, setPanelCollapsed] = useState(false)
  // Zoom (visual) sobre la imagen central
  const [zoom, setZoom] = useState(1)
  // Visor de instrucciones (modal)
  const [showHelp, setShowHelp] = useState(false)

  const imgRef = useRef(null)
  const fileInput = useRef(null)

  // Cargar catálogo de backgrounds locales del backend
  useEffect(() => {
    fetch(`${API_URL}/backgrounds/`)
      .then(r => r.json())
      .then(data => {
        setBackgrounds(data || {})
        const firstCat = Object.keys(data || {})[0]
        const firstBg = firstCat && data[firstCat]?.[0]
        if (firstBg) setSelectedBg(firstBg.id)
      })
      .catch(() => { })
  }, [])

  // Adaptar la posición por defecto según el modo activo
  useEffect(() => {
    if (composeMode === 'studio') {
      setPlacement(p => ({ ...p, x: 0.5, y: 0.78, scale: p.scale || 0.62 }))
    } else {
      setPlacement(p => ({ ...p, x: 0.5, y: 0.72, scale: p.scale || 0.40 }))
    }
  }, [composeMode])

  // Cierra el visor de instrucciones con la tecla ESC
  useEffect(() => {
    if (!showHelp) return
    const onKey = (e) => { if (e.key === 'Escape') setShowHelp(false) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [showHelp])

  // Imagen mostrada en el centro: en segment manual siempre la original (para poder
  // dibujar el bbox), si no, prioriza generado > segmentado > original.
  // En modo 'generate' la entrada original se trata como ya segmentada (checker pattern).
  const centerImage = useMemo(() => {
    if (activeTab === 'segment' && segMode === 'manual' && imageObj) {
      return { url: imageObj.url, kind: 'original' }
    }
    if (genResult) return { url: genResult, kind: 'generated' }
    if (segResult) return { url: segResult, kind: 'segmented' }
    if (imageObj) return { url: imageObj.url, kind: mode === 'generate' ? 'segmented' : 'original' }
    return null
  }, [imageObj, segResult, genResult, activeTab, segMode, mode])

  // En modo generate, la entrada YA cuenta como "segmentada"
  const segmentDone = mode === 'generate' ? !!segBlob : !!segResult

  const generateUnlocked = mode === 'generate'
    ? !!segBlob
    : segmentDone

  const isProcessing = loading || generating

  const getImageBounds = useCallback(() => {
    if (!imgRef.current || !imageObj) return null
    const rect = imgRef.current.getBoundingClientRect()
    const elAspect = rect.width / rect.height
    const imgAspect = imageObj.width / imageObj.height

    let imgLeft, imgTop, imgW, imgH
    if (imgAspect > elAspect) {
      imgW = rect.width
      imgH = rect.width / imgAspect
      imgLeft = rect.left
      imgTop = rect.top + (rect.height - imgH) / 2
    } else {
      imgH = rect.height
      imgW = rect.height * imgAspect
      imgTop = rect.top
      imgLeft = rect.left + (rect.width - imgW) / 2
    }
    return { left: imgLeft, top: imgTop, width: imgW, height: imgH }
  }, [imageObj])

  const screenToRel = useCallback((clientX, clientY) => {
    const b = getImageBounds()
    if (!b) return null
    const rx = Math.max(0, Math.min(1, (clientX - b.left) / b.width))
    const ry = Math.max(0, Math.min(1, (clientY - b.top) / b.height))
    return { rx, ry }
  }, [getImageBounds])

  const resetAllState = useCallback((opts = {}) => {
    setBbox(null)
    setDrawing(null)
    setSegResult(null)
    setSegBlob(null)
    setGenResult(null)
    setError(null)
    setZoom(1)
    if (opts.full) {
      setImageObj(null)
      setImageFile(null)
    }
  }, [])

  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImageFile(file)
    resetAllState()

    if (mode === 'generate') {
      setSegBlob(file)
    }

    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      setImageObj({ url, width: img.naturalWidth, height: img.naturalHeight })
      setResizeMaxSide(pickClosestResize(Math.max(img.naturalWidth, img.naturalHeight)))
    }
    img.src = url
  }, [mode, resetAllState])

  const onFileChange = (e) => loadFile(e.target.files[0])
  const onDrop = (e) => {
    e.preventDefault()
    setFileDragging(false)
    loadFile(e.dataTransfer.files[0])
  }

  // ── Dibujo de bounding box (cuando estamos en segment+manual con imagen cargada) ──
  const drawingEnabled =
    activeTab === 'segment' && segMode === 'manual' && !!imageObj

  const onMouseDown = (e) => {
    if (!drawingEnabled) return
    e.preventDefault()
    const rel = screenToRel(e.clientX, e.clientY)
    if (!rel) return
    setDrawing({ x1: rel.rx, y1: rel.ry, x2: rel.rx, y2: rel.ry })
    setBbox(null)
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

    const left = (bounds.left - wrapperRect.left) + Math.min(box.x1, box.x2) * bounds.width
    const top = (bounds.top - wrapperRect.top) + Math.min(box.y1, box.y2) * bounds.height
    const width = Math.abs(box.x2 - box.x1) * bounds.width
    const height = Math.abs(box.y2 - box.y1) * bounds.height

    return { left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }
  }

  // ── Acciones backend ──
  const handleSegmentBbox = async () => {
    if (!imageFile || !bbox) return
    setLoading(true); setError(null)
    setSegResult(null); setSegBlob(null); setGenResult(null)
    try {
      const fd = new FormData()
      fd.append('file', imageFile)
      fd.append('bbox', JSON.stringify([bbox.x1, bbox.y1, bbox.x2, bbox.y2]))
      if (resizeMaxSide) fd.append('max_side', String(resizeMaxSide))
      const res = await fetch(`${API_URL}/segment_bbox/`, { method: 'POST', body: fd })
      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      setSegBlob(blob)
      setSegResult(URL.createObjectURL(blob))
      if (mode === 'full') setActiveTab('generate')
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  const handleAutoSegment = async () => {
    if (!imageFile) return
    setLoading(true); setError(null)
    setSegResult(null); setSegBlob(null); setGenResult(null)
    try {
      const fd = new FormData()
      fd.append('file', imageFile)
      if (resizeMaxSide) fd.append('max_side', String(resizeMaxSide))
      const res = await fetch(`${API_URL}/segment_auto/`, { method: 'POST', body: fd })
      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      setSegBlob(blob)
      setSegResult(URL.createObjectURL(blob))
      if (mode === 'full') setActiveTab('generate')
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setLoading(false)
    }
  }

  const handleCompose = useCallback(async () => {
    if (!segBlob) return
    if (composeMode === 'scene' && !selectedBg) return
    setGenerating(true); setError(null)
    try {
      const fd = new FormData()
      fd.append('file', segBlob, 'segmentado.png')
      fd.append('position', JSON.stringify([placement.x, placement.y]))
      fd.append('scale', String(placement.scale))

      let url
      if (composeMode === 'studio') {
        fd.append('style', studioStyle)
        url = `${API_URL}/compose/studio/`
      } else {
        fd.append('background_id', selectedBg)
        fd.append('harmonize_strength', String(harmonize))
        url = `${API_URL}/compose/scene/`
      }

      const res = await fetch(url, { method: 'POST', body: fd })
      if (!res.ok) {
        const detail = await res.json().then(d => d.detail).catch(() => res.statusText)
        throw new Error(detail)
      }
      const blob = await res.blob()
      setGenResult(prev => {
        if (prev) URL.revokeObjectURL(prev)
        return URL.createObjectURL(blob)
      })
    } catch (err) {
      setError(err.message || 'Error desconocido')
    } finally {
      setGenerating(false)
    }
  }, [segBlob, composeMode, studioStyle, selectedBg, harmonize, placement])

  // Re-componer automáticamente al cambiar parámetros (con debounce) si ya
  // existe un resultado previo (el usuario está afinando el ajuste).
  useEffect(() => {
    if (!genResult || !segBlob) return
    if (activeTab !== 'generate') return
    const t = setTimeout(() => { handleCompose() }, 250)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [composeMode, studioStyle, selectedBg, harmonize, placement.x, placement.y, placement.scale])

  // ── Acción del botón principal de la esquina inferior derecha ──
  const primaryAction = useMemo(() => {
    if (isProcessing) return { label: 'PROCESANDO…', onClick: null, disabled: true }

    if (activeTab === 'segment') {
      if (!imageObj) return { label: 'SUBE UNA IMAGEN', onClick: null, disabled: true }
      if (segMode === 'auto') {
        return { label: 'EJECUTAR  »', onClick: handleAutoSegment, disabled: false }
      }
      // manual
      if (!bbox) return { label: 'DIBUJA UN BBOX', onClick: null, disabled: true }
      return { label: 'SEGMENTAR  »', onClick: handleSegmentBbox, disabled: false }
    }

    if (activeTab === 'generate') {
      if (!generateUnlocked) return { label: 'SEGMENTA PRIMERO', onClick: null, disabled: true }
      if (composeMode === 'scene' && !selectedBg) return { label: 'ELIGE UN FONDO', onClick: null, disabled: true }
      const label = composeMode === 'studio' ? 'COMPONER ESTUDIO  »' : 'COMPONER ESCENA  »'
      return { label, onClick: handleCompose, disabled: false }
    }

    return { label: '—', onClick: null, disabled: true }
  }, [activeTab, segMode, imageObj, bbox, isProcessing, generateUnlocked, composeMode, selectedBg, handleCompose])

  // ── Tabs disponibles según modo ──
  const showSegmentTab = mode !== 'generate'
  const showGenerateTab = mode !== 'segment'

  const headerTitle =
    mode === 'full' ? 'PROFESIONALIZAR IMAGEN' :
      mode === 'generate' ? 'GENERAR FONDO' :
        'SEGMENTAR'

  const handleReset = () => { resetAllState() }
  const handleClear = () => { resetAllState({ full: true }); setActiveTab(initialTab); setSegMode('auto') }
  const handleZoomIn = () => setZoom(z => Math.min(z + 0.15, 3))
  const handleZoomOut = () => setZoom(z => Math.max(z - 0.15, 0.4))

  return (
    <div className="pipeline">
      <div className="pipeline-grid-overlay" aria-hidden="true" />

      {/* Esquinas decorativas */}
      <span className="corner corner-tl" aria-hidden="true" />
      <span className="corner corner-tr" aria-hidden="true" />
      <span className="corner corner-bl" aria-hidden="true" />
      <span className="corner corner-br" aria-hidden="true" />

      {/* Botón flecha atrás (vuelve al home) */}
      <button className="pl-back" onClick={onBackHome} title="Volver al inicio">
        <span className="pl-back-arrow" aria-hidden="true">←</span>
        <span className="pl-back-label">INICIO</span>
      </button>

      {/* Botón cerrar (vuelve al home) */}
      <button className="pl-close" onClick={onBackHome} title="Volver al inicio">×</button>

      {/* Toolbar superior */}
      <div className="pl-toolbar">
        <button className="tb-btn" onClick={handleReset} title="Reiniciar">↻</button>
        <button className="tb-btn" onClick={handleReset} title="Deshacer">↶</button>
        <button className="tb-btn" onClick={() => setShowHelp(true)} title="Instrucciones">?</button>
        <button className="tb-btn" title="Rehacer">↷</button>
        <span className="tb-title">{headerTitle}</span>
      </div>

      {/* Sidebar izquierda: thumbnail + propiedades */}
      <aside className="pl-sidebar">
        <div className="pl-thumb-card">
          {imageObj ? (
            <>
              <img src={imageObj.url} alt="thumb" className="pl-thumb-img" />
              <button
                className="pl-thumb-close"
                onClick={handleClear}
                title="Quitar imagen"
              >×</button>
            </>
          ) : (
            <button
              className="pl-thumb-empty"
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setFileDragging(true) }}
              onDragLeave={() => setFileDragging(false)}
              onDrop={onDrop}
            >
              <span className="plus">+</span>
              <span className="lbl">SUBIR IMAGEN</span>
            </button>
          )}
        </div>

        <div className="pl-props-card">
          <div className="pl-props-title">PROPIEDADES</div>
          <div className="pl-props-list">
            <PropRow k="ARCHIVO" v={imageFile?.name || '—'} mono />
            <PropRow k="FORMATO" v={shortType(imageFile?.type)} />
            <PropRow
              k="DIMENSIONES"
              v={imageObj ? `${imageObj.width} × ${imageObj.height}` : '—'}
            />
            <PropRow k="TAMAÑO" v={formatBytes(imageFile?.size)} />
            <PropRow k="MODO" v={mode === 'full' ? 'Pipeline' : mode === 'segment' ? 'Segmentar' : 'Generar fondo'} />
            <PropRow k="ESTADO" v={
              isProcessing ? (loading ? 'Segmentando…' : 'Generando…') :
                genResult ? 'Imagen generada' :
                  segResult ? 'Segmentación lista' :
                    imageObj ? 'Listo para procesar' :
                      'Sin imagen'
            } accent={!!(genResult || segResult)} />
          </div>
        </div>

        {error && (
          <div className="pl-error-card">
            <strong>Error:</strong> {error}
          </div>
        )}
      </aside>

      {/* Área central: imagen / placeholder */}
      <main
        className={`pl-stage ${fileDragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setFileDragging(true) }}
        onDragLeave={() => setFileDragging(false)}
        onDrop={onDrop}
      >
        {!imageObj && !isProcessing && (
          <button
            className="pl-stage-drop"
            onClick={() => fileInput.current?.click()}
          >
            <span className="drop-plus">+</span>
            <span>Arrastra una imagen aquí o haz clic para subirla</span>
          </button>
        )}

        {isProcessing && (
          <div className="pl-stage-loading">
            <span className="spinner large" />
            <p>{generating ? 'Generando fondo…' : 'Segmentando…'}</p>
          </div>
        )}

        {centerImage && !isProcessing && (
          <div
            className={`pl-stage-canvas ${centerImage.kind} ${drawingEnabled ? 'draw-cursor' : ''}`}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={() => drawing && setDrawing(null)}
            style={{ '--zoom': zoom }}
          >
            <img
              ref={imgRef}
              src={centerImage.url}
              alt={centerImage.kind}
              className="pl-stage-img"
              draggable={false}
            />
            {(drawing || (bbox && drawingEnabled)) && (() => {
              const style = getRectStyle(drawing || bbox)
              return style ? (
                <div className={`bbox-overlay ${drawing ? 'drawing' : 'done'}`} style={style} />
              ) : null
            })()}
          </div>
        )}

        {/* Controles de zoom (decorativos + funcionales) */}
        <div className="pl-zoom-controls">
          <button className="zoom-btn" onClick={handleZoomIn} title="Acercar">+</button>
          <button className="zoom-btn" onClick={handleZoomOut} title="Alejar">−</button>
        </div>

        {/* Botón descargar */}
        {(genResult || (segResult && mode === 'segment')) && !isProcessing && (
          <a
            className="pl-download"
            href={genResult || segResult}
            download={genResult ? 'generado.png' : 'segmentado.png'}
          >
            DESCARGAR PNG
          </a>
        )}
      </main>

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        hidden
        onChange={onFileChange}
      />

      {/* Panel inferior */}
      <section className="pl-panel">
        <div className={`pl-panel-collapsible ${panelCollapsed ? 'collapsed' : ''}`}>
          {/* Fila 1: sub-opciones del tab activo */}
          <div className="pl-panel-options">
            {activeTab === 'segment' && (
              <div className="opt-seg-row">
                {segMode === 'auto' && (
                  <span className="opt-info">
                    Detecta automáticamente el objeto principal con FastSAM.
                  </span>
                )}
                {segMode === 'manual' && (
                  <span className="opt-info">
                    {imageObj
                      ? (bbox
                        ? 'Recuadro listo. Pulsa SEGMENTAR para procesar (o redibújalo).'
                        : 'Dibuja un recuadro sobre el objeto en la imagen central.')
                      : 'Sube primero una imagen.'}
                  </span>
                )}
                <label className="resize-ctrl">
                  <span className="resize-label">RESOLUCIÓN</span>
                  <select
                    className="resize-select"
                    value={resizeMaxSide ?? ''}
                    onChange={e => setResizeMaxSide(e.target.value ? Number(e.target.value) : null)}
                  >
                    <option value="">Original</option>
                    <option value="3840">4K (3840 px)</option>
                    <option value="2560">2.5K (2560 px)</option>
                    <option value="1920">Full HD (1920 px)</option>
                    <option value="1280">HD (1280 px)</option>
                    <option value="960">960 px</option>
                    <option value="640">640 px</option>
                  </select>
                </label>
              </div>
            )}

            {activeTab === 'generate' && composeMode === 'studio' && (
              <div className="opt-styles">
                {STUDIO_STYLES.map(s => (
                  <button
                    key={s.key}
                    className={`style-card ${studioStyle === s.key ? 'selected' : ''}`}
                    onClick={() => setStudioStyle(s.key)}
                    disabled={!generateUnlocked}
                    title={s.desc}
                  >
                    <div className={`style-preview style-${s.key}`} />
                    <span className="style-label">{s.label}</span>
                  </button>
                ))}
                <ComposeControls
                  placement={placement}
                  onPlacement={setPlacement}
                  showHarmonize={false}
                />
              </div>
            )}

            {activeTab === 'generate' && composeMode === 'scene' && (
              <div className="opt-scene">
                {Object.keys(backgrounds).length === 0 ? (
                  <span className="opt-info">
                    No hay backgrounds en <code>backend/app/backgrounds/</code>. Añade imágenes y reinicia el servidor (o POST <code>/backgrounds/rescan/</code>).
                  </span>
                ) : (
                  <div className="bg-categories">
                    {Object.entries(backgrounds).map(([cat, items]) => (
                      <div key={cat} className="bg-category">
                        <div className="bg-category-title">{cat.toUpperCase()}</div>
                        <div className="bg-grid">
                          {items.map(bg => (
                            <button
                              key={bg.id}
                              className={`bg-thumb ${selectedBg === bg.id ? 'selected' : ''}`}
                              onClick={() => setSelectedBg(bg.id)}
                              disabled={!generateUnlocked}
                              title={bg.label}
                            >
                              <img src={`${API_URL}${bg.thumb_url}`} alt={bg.label} />
                              {bg.reflective && <span className="bg-badge">✨</span>}
                              <span className="bg-thumb-label">{bg.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <ComposeControls
                  placement={placement}
                  onPlacement={setPlacement}
                  showHarmonize={true}
                  harmonize={harmonize}
                  onHarmonize={setHarmonize}
                />
              </div>
            )}
          </div>

          {/* Fila 2: sub-tabs (categorías) */}
          <div className="pl-panel-subtabs">
            {activeTab === 'segment' && (
              <>
                <button
                  className={`subtab ${segMode === 'auto' ? 'active' : ''}`}
                  onClick={() => setSegMode('auto')}
                >AUTO-SEGMENTAR</button>
                <button
                  className={`subtab ${segMode === 'manual' ? 'active' : ''}`}
                  onClick={() => setSegMode('manual')}
                >BOUNDING BOX MANUAL</button>
              </>
            )}
            {activeTab === 'generate' && (
              <>
                <button
                  className={`subtab ${composeMode === 'studio' ? 'active' : ''}`}
                  onClick={() => setComposeMode('studio')}
                >ESTUDIO PROFESIONAL</button>
                <button
                  className={`subtab ${composeMode === 'scene' ? 'active' : ''}`}
                  onClick={() => setComposeMode('scene')}
                >ESCENA COMERCIAL</button>
              </>
            )}
          </div>
        </div>{/* /pl-panel-collapsible */}

        {/* Fila 3: tabs principales + acción */}
        <div className="pl-panel-main">
          <div className="pl-panel-tabs">
            {showSegmentTab && (
              <button
                className={`mtab ${activeTab === 'segment' ? 'active' : ''}`}
                onClick={() => setActiveTab('segment')}
              >
                <span className="mtab-icon">◫</span>
                <span>SEGMENTAR</span>
              </button>
            )}
            {showGenerateTab && (
              <button
                className={`mtab ${activeTab === 'generate' ? 'active' : ''}`}
                onClick={() => setActiveTab('generate')}
              >
                <span className="mtab-icon">✦</span>
                <span>GENERAR FONDO</span>
              </button>
            )}
          </div>

          <button
            className="pl-panel-toggle"
            onClick={() => setPanelCollapsed(c => !c)}
            title={panelCollapsed ? 'Expandir panel' : 'Colapsar panel'}
          >
            {panelCollapsed ? '▲' : '▼'}
          </button>

          <button
            className="pl-action-btn"
            onClick={primaryAction.onClick || (() => { })}
            disabled={primaryAction.disabled}
          >
            {primaryAction.label}
          </button>
        </div>
      </section>

      {/* Visor de instrucciones (modal) */}
      {showHelp && (
        <div
          className="pl-help-backdrop"
          onClick={() => setShowHelp(false)}
          role="dialog"
          aria-modal="true"
        >
          <div className="pl-help-modal" onClick={(e) => e.stopPropagation()}>
            <div className="pl-help-header">
              <span className="pl-help-title">INSTRUCCIONES</span>
              <button
                className="pl-help-close"
                onClick={() => setShowHelp(false)}
                title="Cerrar"
              >×</button>
            </div>

            <div className="pl-help-body">
              {INSTRUCTION_IMAGES.map((src, i) => (
                <HelpFigure key={src} src={src} index={i} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function HelpFigure({ src, index }) {
  const [missing, setMissing] = useState(false)
  return (
    <figure className="pl-help-figure">
      {missing ? (
        <div className="pl-help-missing">
          <span className="pl-help-missing-icon">⊘</span>
          <span>Imagen pendiente</span>
          <code>{src}</code>
        </div>
      ) : (
        <img
          src={src}
          alt={`Instrucción ${index + 1}`}
          onError={() => setMissing(true)}
        />
      )}
      <figcaption>PASO {index + 1}</figcaption>
    </figure>
  )
}

function PropRow({ k, v, mono = false, accent = false }) {
  return (
    <div className="prop-row">
      <span className="prop-k">{k}</span>
      <span className={`prop-v ${mono ? 'mono' : ''} ${accent ? 'accent' : ''}`} title={typeof v === 'string' ? v : ''}>
        {v}
      </span>
    </div>
  )
}

function ComposeControls({ placement, onPlacement, showHarmonize, harmonize, onHarmonize }) {
  return (
    <div className="compose-controls">
      <label className="ctrl">
        <span>Escala objeto</span>
        <input
          type="range" min="0.10" max="1.00" step="0.01"
          value={placement.scale}
          onChange={(e) => onPlacement({ ...placement, scale: parseFloat(e.target.value) })}
        />
        <span className="ctrl-val">{Math.round(placement.scale * 100)}%</span>
      </label>
      <label className="ctrl">
        <span>Posición X</span>
        <input
          type="range" min="0.00" max="1.00" step="0.01"
          value={placement.x}
          onChange={(e) => onPlacement({ ...placement, x: parseFloat(e.target.value) })}
        />
        <span className="ctrl-val">{Math.round(placement.x * 100)}%</span>
      </label>
      <label className="ctrl">
        <span>Pies (Y)</span>
        <input
          type="range" min="0.20" max="0.98" step="0.01"
          value={placement.y}
          onChange={(e) => onPlacement({ ...placement, y: parseFloat(e.target.value) })}
        />
        <span className="ctrl-val">{Math.round(placement.y * 100)}%</span>
      </label>
      {showHarmonize && (
        <label className="ctrl">
          <span>Harmonización</span>
          <input
            type="range" min="0.00" max="1.00" step="0.05"
            value={harmonize}
            onChange={(e) => onHarmonize(parseFloat(e.target.value))}
          />
          <span className="ctrl-val">{Math.round(harmonize * 100)}%</span>
        </label>
      )}
    </div>
  )
}
