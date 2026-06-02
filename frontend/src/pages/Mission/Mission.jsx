import { useNavigate } from 'react-router-dom'
import NavBar from '../../components/NavBar/NavBar.jsx'
import '../Home/Home.css'
import './Mission.css'

export default function Mission() {
  const navigate = useNavigate()

  return (
    <div className="home">
      <div className="home-grid-overlay" aria-hidden="true" />

      <span className="corner corner-tl" aria-hidden="true" />
      <span className="corner corner-tr" aria-hidden="true" />
      <span className="corner corner-bl" aria-hidden="true" />
      <span className="corner corner-br" aria-hidden="true" />

      <NavBar />

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
          ← VOLVER AL INICIO
        </button>
      </div>
    </div>
  )
}
