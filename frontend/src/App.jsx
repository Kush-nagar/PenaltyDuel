import { BrowserRouter, Routes, Route } from 'react-router-dom'
import ScanlineOverlay from './components/ScanlineOverlay'
import TopNav from './components/TopNav'
import Home from './pages/Home'
import Prediction from './pages/Prediction'
import Players from './pages/Players'

export default function App() {
  return (
    <BrowserRouter>
      <ScanlineOverlay />
      <TopNav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/predict" element={<Prediction />} />
        <Route path="/players" element={<Players />} />
      </Routes>
    </BrowserRouter>
  )
}
