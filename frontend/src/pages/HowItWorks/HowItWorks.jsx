import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../Home/Home.css'
import './HowItWorks.css'

const INSTRUCTION_IMAGES = [
  '/instructions/step-1.png',
  '/instructions/step-2.png',
]

function HelpFigure({ src, index }) {
  const [missing, setMissing] = useState(false)
  return (
    <figure className="hiw-help-figure">
      {missing ? (
        <div className="hiw-help-missing">
          <span>⊘</span>
          <span>Imagen pendiente</span>
          <code>{src}</code>
        </div>
      ) : (
        <img
          src={src}
          alt={`Instrucción paso ${index + 1}`}
          onError={() => setMissing(true)}
        />
      )}
    </figure>
  )
}

export default function HowItWorks() {
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const [showHelp, setShowHelp] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!menuOpen) return
    const onClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [menuOpen])

  useEffect(() => {
    if (!showHelp) return
    const onKey = (e) => { if (e.key === 'Escape') setShowHelp(false) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [showHelp])

  const go = (path) => {
    setMenuOpen(false)
    navigate(path)
  }

  return (
    <div className="home">
      <div className="home-grid-overlay" aria-hidden="true" />

      <span className="corner corner-tl" aria-hidden="true" />
      <span className="corner corner-tr" aria-hidden="true" />
      <span className="corner corner-bl" aria-hidden="true" />
      <span className="corner corner-br" aria-hidden="true" />

      <header className="home-header">
        <div className="menu-wrapper" ref={menuRef}>
          <button
            className={`menu-btn ${menuOpen ? 'open' : ''}`}
            onClick={() => setMenuOpen(o => !o)}
            aria-expanded={menuOpen}
          >
            <span className="menu-icon" aria-hidden="true">
              <span /><span /><span />
            </span>
            MENU
          </button>

          {menuOpen && (
            <nav className="menu-panel">
              <ul>
                <li className="menu-item" onClick={() => go('/profesionalizar')}>Profesionalizar imagen</li>
                <li className="menu-item" onClick={() => go('/galeria')}>Galería de estilos</li>
                <li className="menu-item" onClick={() => go('/mision')}>Misión y visión</li>
                <li className="menu-item" onClick={() => go('/como-funciona')}>¿Cómo funciona?</li>
                <li className="menu-item" onClick={() => go('/contacto')}>Contacto</li>
              </ul>
            </nav>
          )}
        </div>
      </header>

      <section className="howitworks-content">
        <h1 className="howitworks-title">¿Cómo funciona?</h1>

        <div className="howitworks-images">
          <img src="/demonstration/joyero_resized_more.jpg" alt="" className="howitworks-img" />
          <img src="/demonstration/joyero_resultado.png" alt="" className="howitworks-img" />
        </div>

        <ol className="howitworks-text howitworks-steps">
          <li>
            <strong>Accede a «Profesionalizar imagen»</strong> desde el menú superior.
            Allí encontrarás el flujo completo en un solo lugar.
          </li>
          <li>
            <strong>Sube tu foto de producto.</strong> Admite cualquier imagen habitual
            de catálogo: fondo neutro, fondo de tienda o incluso una foto tomada
            con el móvil.
          </li>
          <li>
            <strong>Revisa la segmentación.</strong> La herramienta detecta y recorta
            automáticamente el objeto principal. Comprueba que el contorno sea correcto
            antes de continuar.
          </li>
          <li>
            <strong>Elige un estilo de fondo.</strong> Selecciona entre los fondos
            profesionales disponibles: estudio limpio, entorno lifestyle, gradiente
            de marca y más.
          </li>
          <li>
            <strong>Descarga el resultado.</strong> Obtendrás una imagen lista para
            publicar en cualquier marketplace, tienda online o red social, sin
            necesidad de un editor externo.
          </li>
        </ol>

        <div className="howitworks-recommendations">
          <div className="hiw-rec-header">
            <h2 className="hiw-rec-title">Recomendaciones</h2>
            <button
              className="hiw-help-btn"
              onClick={() => setShowHelp(true)}
              title="Ver instrucciones visuales"
            >?</button>
          </div>

          <ul className="hiw-rec-list">
            <li>El objeto debe aparecer <strong>completo</strong> en la fotografía, sin que ninguna parte quede cortada por los bordes.</li>
            <li>Usa preferiblemente un <strong>fondo simple y neutro</strong> (blanco, gris o liso): facilita enormemente la detección automática.</li>
            <li>Asegúrate de que el producto esté <strong>bien iluminado</strong>, sin sombras duras ni zonas sobreexpuestas.</li>
            <li>Evita que aparezcan <strong>otros objetos</strong> en primer plano que puedan confundir a la herramienta.</li>
            <li>
              Si la herramienta no encuentra el objeto o lo segmenta incorrectamente, puedes usar la{' '}
              <strong>Segmentación manual</strong>: dibuja un recuadro alrededor del producto para indicarle exactamente dónde está y facilitar su trabajo.
            </li>
          </ul>
        </div>
      </section>

      <div className="mission-back-wrapper">
        <button className="mission-back-btn" onClick={() => navigate('/')}>
          VOLVER AL INICIO
        </button>
      </div>

      {showHelp && (
        <div
          className="hiw-help-backdrop"
          onClick={() => setShowHelp(false)}
          role="dialog"
          aria-modal="true"
        >
          <div className="hiw-help-modal" onClick={(e) => e.stopPropagation()}>
            <div className="hiw-help-modal-header">
              <span className="hiw-help-modal-title">INSTRUCCIONES</span>
              <button
                className="hiw-help-close"
                onClick={() => setShowHelp(false)}
                title="Cerrar"
              >×</button>
            </div>
            <div className="hiw-help-modal-body">
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
