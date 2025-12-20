import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './History.css';

interface HistoryItem {
    id: number;
    user_id: string;
    exercise: string;
    rep_count: number;
    video_path: string;
    timestamp: string;
    thumbnail_name: string;
}

const base_URL = 'http://localhost:8080/repCount/download/visualization/';

const fetchUserHistory = async (token: string): Promise<HistoryItem[]> => {

    const endPointUrl = 'http://localhost:8080/users/show_history';

    const response = await fetch(endPointUrl, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
        }
    });

    if (!response.ok) {
          const data = await response.json();
          const errorMessage = data.detail || 'Error al descarregar el vídeo.';
          throw new Error(errorMessage);
        }

    const data: HistoryItem[] = await response.json();
    return data;
};

function History(): React.ReactElement {
  const navigate = useNavigate();

  const [error, setError] = useState<string>('');
  const [message, setMessage] = useState<string>('');

  const [historyList, setHistoryList] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [username, setUsername] = useState<string>('');
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);

  const token = localStorage.getItem('access_token');
  const storedUser = localStorage.getItem('username');

  // Comprova si l'usuari està loguejat al entrar a la web
  useEffect(() => {

  if (token) {
    setIsLoggedIn(true);
    setUsername(storedUser || 'Usuari');
    setError('');
  } else {
    setIsLoggedIn(false);
    setUsername('');
  } }, [token, storedUser]);

  // Mostrar l'historial
  
    useEffect(() => {
      if (isLoggedIn && token) {
        setIsLoading(true);
        fetchUserHistory(token)
            .then((data) => {
                setHistoryList(data);
                setIsLoading(false);
            })
            .catch(() => {
                setError('Error en carregar l\'historial.');
            })
            .finally(() => {
                setIsLoading(false);
            });
      }
    }, [isLoggedIn, token]);

  // Gestió de la caducitat del token
  useEffect(() => {
    const expiryString = localStorage.getItem('token_expiry');
    const token = localStorage.getItem('access_token');

    if (!token || !expiryString) {
      setIsLoggedIn(false);
      setUsername('');
      return;
    }

    const expiryTime = parseInt(expiryString, 10);
    const timeRemaining = expiryTime - Date.now();

    if (timeRemaining <= 0) {
      forceLogOut();
      return;
    }

    const timeoutId = setTimeout(() => {
      console.log("La sessió ha caducat. Torna a iniciar sessió.")
      forceLogOut();
    }, timeRemaining);

    return () => clearTimeout(timeoutId);

  }, [isLoggedIn, navigate]);

  //Tanca sessió quan el token ha caducat
  const forceLogOut = () => {
    localStorage.clear();
    setIsLoggedIn(false);
    setUsername('');
    setMessage('La sessió ha caducat. Si us plau, inicia sessió de nou.');
  }

  return (
    <div className="card-container history-page">
      <button onClick={() => navigate('/')} className="back-button">
              ← Pàgina principal
      </button>

      <div className="history-header">
        <h1 className="gradient-title">Historial de {username}</h1>          
      </div>

      {message && <p className="message-text"> {message} </p>}
      {error && <p className="error-message" style={{color: 'red'}}>{error}</p>}


        <div className="history-content" style={{color: 'black'}}>
          {isLoading ? (
            <div className="loading-spinner">Carregant historial...</div>
          ) : (
            <div className="history-list-container">
                {historyList.length > 0 ? (
                    <ul className="history-list">
                        {historyList.map((item) => (
                            <li key={item.id} className="history-item">
                              <img src={`${base_URL}${item.thumbnail_name}/image?token=${token}`} alt="Thumbnail" style={{maxWidth: '250px', maxHeight: '250px'}} />
                                <div className="history-date-box">
                                    <span className="history-date">{new Date(item.timestamp).toLocaleString()}</span>
                                </div>
                                <div className="history-details">
                                    <h3>{item.exercise}</h3>
                                    <p>Repeticions: {item.rep_count}</p>
                                </div>
                            </li>
                        ))}
                    </ul>
                ) :(<p> No hi ha entrades a l'historial.</p>
                    
                )}
            </div>         
          )}
        </div>

    </div>
  );
}
export default History;