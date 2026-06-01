import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import './Home.css'

// ── Imágenes (rellena con tus rutas) ─────────────────────────────────
// Coloca tus archivos en /frontend/public/ y referencia con "/archivo.jpg"
const HERO_IMAGE_URL = ''   // p.ej. '/dress.png'

// ── Textos editables ─────────────────────────────────────────────────
const BRAND_LOGO = 'Professional Product Studio'
const BRAND_TITLE = 'ProdStudio'
const BRAND_TAGLINE = 'DE FOTOGRAFÍA A IMAGEN PROFESIONAL'
const HERO_LEAD = ['PROFESIONALIZA', 'UNA IMAGEN', 'FÁCILMENTE', 'EN SEGUNDOS']
const COORD_LEFT = '37.3881° N'
const COORD_RIGHT = '-5.9953° W'

export default function Home() {
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
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

  const go = (path) => {
    setMenuOpen(false)
    navigate(path)
  }

  const heroStyle = {
    backgroundImage: HERO_IMAGE_URL ? `url(${HERO_IMAGE_URL})` : 'none',
  }

  return (
    <div className="home">
      <div className="home-grid-overlay" aria-hidden="true" />

      <video
        className="home-bg-video"
        src="/video/video_loop.mp4"
        autoPlay
        loop
        muted
        playsInline
        aria-hidden="true"
      />

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

          <nav className={`menu-panel${menuOpen ? ' open' : ''}`}>
              <ul>
                <li className="menu-item" onClick={() => go('/profesionalizar')}>Profesionalizar imagen</li>
                <li className="menu-item" onClick={() => go('/galeria')}>Galería de estilos</li>
                <li className="menu-item" onClick={() => go('/mision')}>Misión y visión</li>
                <li className="menu-item" onClick={() => go('/como-funciona')}>¿Cómo funciona?</li>
                <li className="menu-item" onClick={() => go('/contacto')}>Contacto</li>
              </ul>
            </nav>
        </div>

        <div className="brand-logo">{BRAND_LOGO}</div>

        <button className="cta-3d" onClick={() => go('/como-funciona')}>
          CONOCE EL PROYECTO <span className="cta-icon" aria-hidden="true">⌖</span>
        </button>
      </header>

      <div className="hero-lead">
        {HERO_LEAD.map((line, i) => <span key={i}>{line}</span>)}
      </div>

      <div className="plus-marker" aria-hidden="true">+ + +</div>

      <div className="hero-image" style={heroStyle} aria-hidden={!HERO_IMAGE_URL} />

      <div className="brand-title-wrapper">
        <h1 className="brand-title">{BRAND_TITLE}</h1>
        <p className="brand-tagline">{BRAND_TAGLINE}</p>
      </div>

      <span className="coord coord-left">{COORD_LEFT}</span>
      <span className="coord coord-right">{COORD_RIGHT}</span>
    </div>
  )
}
