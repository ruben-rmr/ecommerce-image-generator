import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../Home/Home.css'
import './Mission.css'

export default function Mission() {
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const [submenuOpen, setSubmenuOpen] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    if (!menuOpen) return
    const onClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false)
        setSubmenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [menuOpen])

  const go = (path) => {
    setMenuOpen(false)
    setSubmenuOpen(false)
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
            onClick={() => { setMenuOpen(o => !o); setSubmenuOpen(false) }}
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
                <li
                  className={`menu-item has-submenu ${submenuOpen ? 'sub-open' : ''}`}
                  onMouseEnter={() => setSubmenuOpen(true)}
                  onMouseLeave={() => setSubmenuOpen(false)}
                >
                  <span className="menu-item-label" onClick={() => go('/profesionalizar')}>
                    Profesionalizar imagen
                    <span className="chev" aria-hidden="true">›</span>
                  </span>
                  <ul className="submenu">
                    <li onClick={() => go('/segmentar')}>Segmentar</li>
                    <li onClick={() => go('/generar-fondo')}>Generar fondo</li>
                  </ul>
                </li>
                <li className="menu-item" onClick={() => go('/galeria')}>Galería de estilos</li>
                <li className="menu-item" onClick={() => go('/mision')}>Misión y visión</li>
                <li className="menu-item" onClick={() => go('/como-funciona')}>¿Cómo funciona?</li>
                <li className="menu-item" onClick={() => go('/contacto')}>Contacto</li>
              </ul>
            </nav>
          )}
        </div>
      </header>

      <section className="mission-content">
        <div className="mission-col">
          <h2>Misión</h2>
          <p>
            Ayudar a pequeñas y medianas marcas de e-commerce a transformar
            fotografías cotidianas en imágenes de producto profesionales,
            haciendo accesible una calidad visual antes reservada a grandes
            estudios. Queremos reducir el tiempo, el coste y la fricción
            técnica de producir contenido visual de alto impacto.
          </p>
        </div>
        <div className="mission-col">
          <h2>Visión</h2>
          <p>
            Convertirnos en el estudio de producto virtual de referencia:
            un espacio donde cualquier creador pueda generar, en segundos,
            imágenes coherentes con la identidad de su marca. Aspiramos a
            redefinir la fotografía de producto combinando inteligencia
            artificial, diseño y experiencia de usuario.
          </p>
        </div>
      </section>

      <div className="mission-back-wrapper">
        <button className="mission-back-btn" onClick={() => navigate('/')}>
          VOLVER AL INICIO
        </button>
      </div>
    </div>
  )
}
