import NavBar from '../../components/NavBar/NavBar.jsx'
import './Home.css'

// ── Imágenes (rellena con tus rutas) ─────────────────────────────────
// Coloca tus archivos en /frontend/public/ y referencia con "/archivo.jpg"
const HERO_IMAGE_URL = ''   // p.ej. '/dress.png'

// ── Textos editables ─────────────────────────────────────────────────
const BRAND_TITLE = 'Photify'
const BRAND_TAGLINE = 'DE FOTO A IMAGEN PROFESIONAL'
const HERO_LEAD = ['PROFESIONALIZA', 'UNA IMAGEN', 'FÁCILMENTE', 'EN SEGUNDOS']
const COORD_LEFT = 'TFG 2026'
const COORD_RIGHT = 'Universidad de Sevilla'

export default function Home() {
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

      <NavBar />

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
