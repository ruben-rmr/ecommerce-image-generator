import { Routes, Route, Navigate } from 'react-router-dom'
import Home from './pages/Home/Home.jsx'
import Pipeline from './pages/Pipeline/Pipeline.jsx'
import Gallery from './pages/Gallery/Gallery.jsx'
import Mission from './pages/Mission/Mission.jsx'
import Contact from './pages/Contact/Contact.jsx'
import Preview3D from './pages/Preview3D/Preview3D.jsx'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/profesionalizar" element={<Pipeline mode="full" />} />
      <Route path="/segmentar"       element={<Pipeline mode="segment" />} />
      <Route path="/generar-fondo"   element={<Pipeline mode="generate" />} />
      <Route path="/galeria"   element={<Gallery />} />
      <Route path="/mision"    element={<Mission />} />
      <Route path="/contacto"  element={<Contact />} />
      <Route path="/preview-3d" element={<Preview3D />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
