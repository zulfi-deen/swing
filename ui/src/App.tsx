import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Recommendations from './pages/Recommendations';
import Twins from './pages/Twins';
import Performance from './pages/Performance';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/twins" element={<Twins />} />
        <Route path="/performance" element={<Performance />} />
      </Routes>
    </Router>
  );
}

export default App;


