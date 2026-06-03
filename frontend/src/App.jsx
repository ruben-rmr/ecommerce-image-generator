import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Home from './pages/Home/Home.jsx'
import Pipeline from './pages/Pipeline/Pipeline.jsx'
import Gallery from './pages/Gallery/Gallery.jsx'
import Mission from './pages/Mission/Mission.jsx'
import HowItWorks from './pages/HowItWorks/HowItWorks.jsx'
import LearnMore from './pages/LearnMore/LearnMore.jsx'
export default function App() {
  const location = useLocation()
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/profesionalizar" element={<Pipeline />} />
      <Route path="/galeria" element={<Gallery />} />
      <Route path="/mision" element={<Mission />} />
      <Route path="/como-funciona" element={<HowItWorks />} />
      <Route path="/saber-mas" element={<LearnMore />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
