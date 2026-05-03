import { useNavigate } from 'react-router-dom'

export default function PlaceholderPage({ title }) {
  const navigate = useNavigate()
  return (
    <div style={{
      minHeight: '100vh',
      background: '#0a0820',
      color: '#fff',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '1.2rem',
      padding: '2rem',
      fontFamily: 'system-ui, sans-serif',
    }}>
      <h1 style={{ fontSize: '2rem', letterSpacing: '0.1em' }}>{title}</h1>
      <p style={{ opacity: 0.7 }}>Próximamente.</p>
      <button
        onClick={() => navigate('/')}
        style={{
          padding: '0.6rem 1.2rem',
          background: 'transparent',
          border: '1.5px solid #c8ff2e',
          color: '#c8ff2e',
          borderRadius: 4,
          fontWeight: 700,
          letterSpacing: '0.15em',
          cursor: 'pointer',
        }}
      >
        ← VOLVER AL INICIO
      </button>
    </div>
  )
}
