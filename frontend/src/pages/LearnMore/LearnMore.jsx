import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import NavBar from '../../components/NavBar/NavBar.jsx'
import '../Home/Home.css'
import './LearnMore.css'

const FAQ = [
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
    a: 'La segmentación automática suele completarse en menos de 2 segundos. La implementación con un fondo puede tardar entre 1 y 3 segundos según la complejidad del estilo elegido y la carga del servidor.',
  },
  {
    q: '¿Puedo usar las imágenes generadas con fines comerciales?',
    a: 'Si las imágenes resultantes son de tu propiedad, sí, y puedes utilizarlas libremente en tiendas online, marketplaces, redes sociales y cualquier otro canal comercial.',
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
    a: 'Actualmente los estilos disponibles pueden verse de forma más visual en la "Galería de estilos", accesible desde el menú superior central.',
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

export default function LearnMore() {
  const navigate = useNavigate()
  const [openIndex, setOpenIndex] = useState(null)

  const toggle = (i) => setOpenIndex(prev => (prev === i ? null : i))

  return (
    <div className="home">
      <div className="home-grid-overlay" aria-hidden="true" />

      <span className="corner corner-tl" aria-hidden="true" />
      <span className="corner corner-tr" aria-hidden="true" />
      <span className="corner corner-bl" aria-hidden="true" />
      <span className="corner corner-br" aria-hidden="true" />

      <NavBar />

      <section className="learnmore-content">
        <h1 className="learnmore-title">Saber más: FAQ</h1>

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
