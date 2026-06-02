import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import NavBar from '../../components/NavBar/NavBar.jsx'
import { GALLERY_ITEMS } from './galleryData.js'
import '../Home/Home.css'
import './Gallery.css'

// Ventana de slides visibles a cada lado del activo (las demás quedan ocultas).
const VISIBLE_RANGE = 2

export default function Gallery() {
  const navigate = useNavigate()
  const [active, setActive] = useState(0)
  const count = GALLERY_ITEMS.length

  const go = useCallback((dir) => {
    setActive((prev) => (prev + dir + count) % count)
  }, [count])

  // Navegación con las flechas del teclado
  useEffect(() => {
    if (count < 2) return
    const onKey = (e) => {
      if (e.key === 'ArrowLeft') go(-1)
      else if (e.key === 'ArrowRight') go(1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, count])

  const activeItem = GALLERY_ITEMS[active]

  return (
    <div className="home">
      <div className="home-grid-overlay" aria-hidden="true" />

      <span className="corner corner-tl" aria-hidden="true" />
      <span className="corner corner-tr" aria-hidden="true" />
      <span className="corner corner-bl" aria-hidden="true" />
      <span className="corner corner-br" aria-hidden="true" />

      <NavBar />

      <section className="gallery">
        {count === 0 ? (
          <div className="gallery-empty">
            <span className="gallery-empty-icon" aria-hidden="true">🖼</span>
            <p>Aún no hay imágenes en la galería.</p>
            <p className="gallery-empty-hint">
              Coloca tus archivos en{' '}
              <code>frontend/src/pages/Gallery/images/</code> y aparecerán aquí
              automáticamente.
            </p>
          </div>
        ) : (
          <>
            <div className="gallery-stage">
              <button
                className="gallery-arrow gallery-arrow--left"
                onClick={() => go(-1)}
                aria-label="Imagen anterior"
                disabled={count < 2}
              >
                ‹
              </button>

              <div className="gallery-track">
                {GALLERY_ITEMS.map((item, i) => {
                  // Distancia (con envoltura circular) al slide activo.
                  let offset = i - active
                  if (offset > count / 2) offset -= count
                  if (offset < -count / 2) offset += count
                  const abs = Math.abs(offset)
                  const hidden = abs > VISIBLE_RANGE
                  const isActive = offset === 0

                  return (
                    <figure
                      key={item.src}
                      className={`gallery-slide${isActive ? ' is-active' : ''}`}
                      style={{
                        // Los slides ocultos se aparcan justo fuera de la ventana
                        // visible (sin animar) para evitar barridos en pantalla.
                        '--offset': hidden ? Math.sign(offset) * (VISIBLE_RANGE + 1) : offset,
                        '--abs': hidden ? VISIBLE_RANGE + 1 : abs,
                        opacity: hidden ? 0 : 1,
                        zIndex: count - abs,
                        pointerEvents: isActive || hidden ? 'none' : 'auto',
                        transition: hidden ? 'none' : undefined,
                      }}
                      onClick={() => !hidden && !isActive && setActive(i)}
                      aria-hidden={!isActive}
                    >
                      <img src={item.src} alt={item.title} draggable={false} />
                    </figure>
                  )
                })}
              </div>

              <button
                className="gallery-arrow gallery-arrow--right"
                onClick={() => go(1)}
                aria-label="Imagen siguiente"
                disabled={count < 2}
              >
                ›
              </button>
            </div>

            <div className="gallery-caption" key={active}>
              <h2 className="gallery-title">{activeItem.title}</h2>
              <p className="gallery-desc">{activeItem.description}</p>
            </div>
          </>
        )}
      </section>

      <div className="gallery-back-wrapper">
        <button className="gallery-back-btn" onClick={() => navigate('/')}>
          ← VOLVER AL INICIO
        </button>
      </div>
    </div>
  )
}
