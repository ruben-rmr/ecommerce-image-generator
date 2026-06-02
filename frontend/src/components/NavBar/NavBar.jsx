import { NavLink } from 'react-router-dom'
import './NavBar.css'

// Marca y enlaces de navegación de toda la web. Se define una sola vez aquí y
// se reutiliza en cada página (<NavBar />), en lugar de duplicar el menú.
const BRAND = 'Photify'

const LINKS = [
  { to: '/', label: 'Inicio', end: true },
  { to: '/profesionalizar', label: 'Profesionalizar' },
  { to: '/galeria', label: 'Galería' },
  { to: '/mision', label: 'Misión' },
  { to: '/como-funciona', label: '¿Cómo funciona?' },
  { to: '/contacto', label: 'Contacto' },
]

export default function NavBar() {
  return (
    <header className="navbar">

      <nav className="navbar-nav" aria-label="Navegación principal">
        {LINKS.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `navbar-link${isActive ? ' active' : ''}`}
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}
