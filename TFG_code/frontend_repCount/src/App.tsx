import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RepCount from './RepCount.tsx';
import Register from './Register.tsx';
import Login from './Login.tsx';
import History from './History.tsx';
import UserProfile from './UserProfile.tsx';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<RepCount />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/History" element={<History />} />
        <Route path="/UserProfile" element={<UserProfile />} /> 
      </Routes>
    </Router>
  );
}

export default App;