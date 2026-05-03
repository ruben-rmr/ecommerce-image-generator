import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import '../Home/Home.css'
import './Contact.css'

const FAQ = [
  {
    q: '¿Quién soy?',
    a: 'Mi nombre es Rubén, estudiante de Ingeniería Informática en la Universidad de Sevilla, con gran interés por el diseño full-stack y la integración de la IA en los negocios.',
  },
  {
    q: '¿En qué consiste esta página web?',
    a: 'Esta página web nace de mi trabajo/proyecto de fin de grado universitario (TFG), como demostración de mi aprendizaje y capacidad como ingeniero durante mis años como alumno en la Universidad de Sevilla, en ramas de la informática como el diseño arquitéctonico sistemas de SaaS, desarrollo e implementación de un pipeline que combina front-end y back-end con la integración de modelos de segmentación de imágenes y su tuneo para este caso específico.',
  },
  {
    q: '¿Qué tipo de imágenes puedo subir?',
    a: 'Puedes subir cualquier fotografía de producto en formato JPG, PNG o WEBP. Funciona mejor con imágenes bien iluminadas y fondo neutro, aunque la herramienta también procesa fotos con fondos complejos.',
  },
  {
    q: '¿Cuánto tiempo tarda en generarse el resultado?',
    a: 'La segmentación automática suele completarse en menos de 5 segundos. La generación del fondo puede tardar entre 10 y 30 segundos según la complejidad del estilo elegido y la carga del servidor.',
  },
  {
    q: '¿Puedo usar las imágenes generadas con fines comerciales?',
    a: 'Sí. Las imágenes resultantes son de tu propiedad y puedes utilizarlas libremente en tiendas online, marketplaces, redes sociales y cualquier otro canal comercial.',
  },
  {
    q: '¿Qué hago si la segmentación no detecta bien mi producto?',
    a: 'Usa la opción de Segmentación manual: dibuja un recuadro alrededor del producto para indicarle a la herramienta exactamente dónde se encuentra.',
  },
  {
    q: '¿Se guardan mis imágenes en algún servidor?',
    a: 'Las imágenes se procesan en el servidor únicamente durante la sesión activa y no se almacenan de forma permanente. No compartimos tus archivos con terceros.',
  },
  {
    q: '¿Qué estilos de fondo están disponibles?',
    a: 'Actualmente los estilos disponibles pueden verse de forma más visual en la "Galería de estilos", accesible desde el menú en la esquina superior izquierda.',
  },
  {
    q: '¿Cómo puedo contactar con el desarrollador?',
    a: 'Puedes enviarme un correo a rubromgui@alum.us.es, y te contestaré lo más rápido posible.',
  },
]

function AccordionItem({ item, isOpen, onToggle }) {
  const bodyRef = useRef(null)

  return (
    <li className={`faq-item ${isOpen ? 'faq-item--open' : ''}`}>
      <button
        className="faq-question"
        onClick={onToggle}
        aria-expanded={isOpen}
      >
        <span className="faq-question-text">{item.q}</span>
        <span className="faq-chevron" aria-hidden="true">{isOpen ? '−' : '+'}</span>
      </button>
      <div
        className="faq-answer-wrapper"
        ref={bodyRef}
        style={{ maxHeight: isOpen ? bodyRef.current?.scrollHeight + 'px' : '0px' }}
      >
        <p className="faq-answer">{item.a}</p>
      </div>
    </li>
  )
}

export default function Contact() {
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const [submenuOpen, setSubmenuOpen] = useState(false)
  const [openIndex, setOpenIndex] = useState(null)
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

  const toggle = (i) => setOpenIndex(prev => (prev === i ? null : i))

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

      <section className="contact-content">
        <h1 className="contact-title">Contacto e Información</h1>

        <ul className="faq-list">
          {FAQ.map((item, i) => (
            <AccordionItem
              key={i}
              item={item}
              isOpen={openIndex === i}
              onToggle={() => toggle(i)}
            />
          ))}
        </ul>
      </section>

      <div className="mission-back-wrapper">
        <button className="mission-back-btn" onClick={() => navigate('/')}>
          VOLVER AL INICIO
        </button>
      </div>
    </div>
  )
}
