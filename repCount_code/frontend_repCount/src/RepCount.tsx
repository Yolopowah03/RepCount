import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

type ViewMode = 'VIEW' | 'CLOSING_SESSION';

function RepCount(): React.ReactElement {
  const navigate = useNavigate();

  const [viewMode, setViewMode] = useState<ViewMode>('VIEW');
  const [error, setError] = useState<string>('');

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string>('');
  const [imageName, setImageName] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');

  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [showMenu, setShowMenu] = useState<boolean>(false);
  const [username, setUsername] = useState<string>('');
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);

  const [clientId] = useState<string>(uuidv4());
  const [progress, setProgress] = useState<number>(0);
  const [processingStatus, setProcessingStatus] = useState<string>('');
  const ws = useRef<WebSocket | null>(null);

  const token = localStorage.getItem('access_token');
  const storedUser = localStorage.getItem('username');

  useEffect(() => {
    ws.current = new WebSocket(`ws://your_port1:8080/repCount/ws/progress/${clientId}`);

    setProgress(0);
    setProcessingStatus('');

    ws.current.onopen = () => {
      console.log('WebSocket connection opened');
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.progress !== undefined) {
        setProgress(data.progress);
      }
      if (data.status) {
        setProcessingStatus(data.status);
      }
    };

    return () => {
      ws.current?.close();

    };
  }, [clientId]);

  // Comprova si l'usuari està loguejat al entrar a la web
  useEffect(() => {

    if (token) {
      setIsLoggedIn(true);
      setUsername(storedUser || 'Usuari');
      setTimeout(() => setShowMenu(false), 3000);
      setError('');
    } else {
      setIsLoggedIn(false);
      setUsername('');
    }
  }, []);

  // Genera la vista prèvia del vídeo seleccionat
  useEffect(() => {

    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    if (!selectedFile) return;

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);

      if (processedVideoUrl === objectUrl) setProcessedVideoUrl(null);
    };
  }, [selectedFile]);

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
    setShowMenu(false);
    setMessage('La sessió ha caducat. Si us plau, inicia sessió de nou.');
    switchMode('VIEW');
  }

  //Canvia entre pestanya principal i pestanya de tancar sessió
  const switchMode = (mode: ViewMode) => {
    setViewMode(mode);
  };


  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setMessage('');
      setProcessedVideoUrl(null);
      setError('');
    } else {
      setSelectedFile(null);
      setError('Selecciona un arxiu de vídeo vàlid.');
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setProgress(0);

    if (!selectedFile) {
      setMessage('Selecciona un vídeo per pujar.');
      return;
    }

    const token = localStorage.getItem('access_token');

    if (!token) {
      setIsLoading(false);
      setMessage('Si us plau, inicia sessió per utilitzar la plataforma.');
      return;
    }

    setIsLoading(true);
    setMessage('');
    setProcessedVideoUrl(null);
    setProcessedImageUrl(null);

    const formData = new FormData();
    formData.append('client_id', clientId);
    formData.append('video_file', selectedFile as Blob);

    //TO DO
    formData.append('skip_frames', '1'); // Exemple de paràmetre addicional
    formData.append('vel_reduction', '1.0'); // Exemple de paràmetre addicional

    try {
      const response = await fetch('http://your_port1:8080/repCount/video_processing', {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.status === 401) {
        throw new Error('Si us plau, torna a iniciar sessió per utilitzar la plataforma.');
      }

      if (!response.ok) {
        throw new Error(`Error en la resposta del servidor: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.video_name && data.image_name && data.predicted_exercise !== undefined) {

        const baseURL = 'http://your_port1:8080/repCount/download/visualization/';
        const token = localStorage.getItem('access_token');

        const videoUrl = `${baseURL}${data.video_name}/video?token=${token}`;
        const imageUrl = `${baseURL}${data.image_name}/image?token=${token}`;
        setProcessedVideoUrl(videoUrl);
        setProcessedImageUrl(imageUrl);
        setVideoName(data.video_name);
        setImageName(data.image_name);
        setMessage(`Repeticions comptades: ${data.count}, Exercici: ${data.predicted_exercise}`);
        setError('');
        setProgress(100);
      } else {
        console.error('Resposta del servidor incompleta:', data);
        throw new Error('Resposta inesperada del servidor.');
      }

    } catch (error) {
      console.error('Error al pujar o processar el vídeo:', error);
      if (error instanceof Error) {
        setError(`${error.message}`);
      } else {
        setError('Hi ha hagut un error al processar el vídeo. Torna-ho a intentar més tard.');
      }
    } finally {
      setIsLoading(false);
      setProcessingStatus('');
    }

  };

  const handleLogOut = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('username');
    localStorage.removeItem('full_name');
    localStorage.removeItem('email');
    localStorage.removeItem('token_expiry');
    setIsLoggedIn(false);
    setUsername('');
    setShowMenu(false);
    setMessage('Sessió tancada correctament.');
    switchMode('VIEW');
  }

  const handleDownloadVideo = async () => {

    try {
      const response = await fetch('http://your_port1:8080/repCount/download/button/' + videoName + '/video', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
      );

      if (!response.ok) {
        const data = await response.json();
        const errorMessage = data.detail || 'Error al descarregar el vídeo.';
        throw new Error(errorMessage);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = videoName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Error al descarregar el vídeo:', error);
      if (error instanceof Error) {
        setError(`${error.message}`);
      } else {
        setError('Hi ha hagut un error al processar el vídeo. Torna-ho a intentar més tard.');
      }
    }
  }

  const handleDownloadImage = async () => {

    try {
      const response = await fetch('http://your_port1:8080/repCount/download/button/' + imageName + '/image', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }
      );

      if (!response.ok) {
        const data = await response.json();
        const errorMessage = data.detail || 'Error al descarregar la imatge.';
        throw new Error(errorMessage);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = imageName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

    } catch (error) {
      console.error('Error al descarregar la imatge:', error);
      if (error instanceof Error) {
        setError(`${error.message}`);
      } else {
        setError('Hi ha hagut un error al processar la imatge. Torna-ho a intentar més tard.');
      }
    }
  }

  return (
    <div className="main-wrapper">
      <nav className="top-navbar">
        <div className="user-menu-container">
          {isLoggedIn ? (
            <div
              onMouseEnter={() => setShowMenu(true)}
              onMouseLeave={() => setShowMenu(false)}
            >

              <button
                onClick={() => setShowMenu(!showMenu)}
                className="username-button"
                title="Usuari"
              >
                <span style={{ marginRight: '20px' }}></span>
                {username}
                <span style={{ fontSize: '12px', marginLeft: '5px', marginRight: '50px' }}>▼</span>
              </button>

              {showMenu && (
                <div className="menu">
                  <button
                    className="menu-item"
                    onClick={() => navigate('/UserProfile')}
                  >
                    Opcions
                  </button>
                  <button
                    className="menu-item"
                    onClick={() => navigate('/History')}
                  >
                    Historial
                  </button>
                  <button
                    className="menu-item"
                    onClick={() => switchMode('CLOSING_SESSION')}
                  >
                    Tancar Sessió
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="auth-buttons-container">
              <button
                className="btn-secondary"
                onClick={() => navigate('/login')}
              >
                Inicia Sessió
              </button>
              <button
                className="btn-primary"
                onClick={() => navigate('/register')}
              >
                Registra't
              </button>
            </div>
          )}
        </div>
      </nav>

      <div className="card-container">
        {viewMode === 'VIEW' && (
          <div>
            <h1 className="gradient-title"> RepCount </h1>

            <h2 className="gradient-subtitle"> Reconeixement d'exercici i comptador automàtic de repeticions per Visió per Computador </h2>

            <p className="description-text">
              Puja un video, i et comptarem les repeticions d'exercicis que has fet!
            </p>

            <form onSubmit={handleSubmit} className="form-centered">
              <div className="form-group">
                <label htmlFor="video-upload">
                  Selecciona el teu vídeo:
                </label>
                <input
                  id="video-upload"
                  type="file"
                  accept="video/*"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </div>

              {isLoading ? (
                <div style={{ width: '100%', maxWidth: '300px', margin: '0 auto' }}>
                  <p style={{ marginBottom: '5px', fontSize: '0.9rem', color: '#555' }}>
                    {processingStatus || 'Processant...'} ({progress}%)
                  </p>
                  <div style={{ width: '100%', backgroundColor: '#e0e0e0', borderRadius: '8px', overflow: 'hidden', height: '10px' }} >
                    <div style={{ width: `${progress}%`, backgroundColor: '#4caf50', height: '100%', transition: 'width 0.3s ease-in-out' }} />
                  </div>
                </div>
              ) : (
                <button type="submit"
                  disabled={!selectedFile}
                  className="submit-button">
                  Comptar Repeticions
                </button>
              )}
            </form>

            {!isLoading && message && (
              <p className="message-text">
                {message}
              </p>
            )}

            {(previewUrl || processedVideoUrl || processedImageUrl) && (
              <div className="media-container">
                <div className="video-results">

                  {(previewUrl) && (
                    <div className="video-box">
                      <h2 className="original-title">Vista Previa (Original)</h2>
                      <video
                        key={previewUrl}
                        width="100%"
                        controls
                      >
                        <source src={previewUrl} type={selectedFile?.type || 'video/mp4'} />
                        El teu navegador no soporta el format del vídeo.
                      </video>
                    </div>)}

                  {processedVideoUrl && (
                    <div className="video-output">
                      <div className="video-box">
                        <h2 className="original-title">Vídeo Processat</h2>
                        <video
                          key={processedVideoUrl}
                          width="100%"
                          controls
                          autoPlay
                          muted
                        >
                          <source src={processedVideoUrl} type={"video/mp4"} />
                          El teu navegador no suporta el format del vídeo.
                        </video>
                      </div>
                      <button
                        onClick={() => handleDownloadVideo()}
                        style={{ fontSize: '0.9rem', padding: '5px 10px' }}
                        className="submit-button"
                      >
                        Descarrega Vídeo
                      </button>
                    </div>
                  )}

                  {processedImageUrl && (
                    <div className="image-output">
                      <h3>Anàlisi de la repetició</h3>
                      <img
                        src={processedImageUrl}
                        alt="Anàlisi de la repetició"
                        width="100%"
                        style={{ borderRadius: '8px', border: '1px solid #ddd' }}
                      />
                      <button
                        onClick={() => handleDownloadImage()}
                        className="submit-button" style={{ fontSize: '0.9rem', padding: '5px 10px' }}
                      >
                        Descarrega Imatge
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

        )}

        {viewMode === 'CLOSING_SESSION' && (
          <div style={{ textAlign: 'center' }}>
            <p>Vols tancar sessió?</p>

            <button className="submit-button logout-button"
              onClick={handleLogOut}
            >
              Tancar Sessió
            </button>

            <button onClick={() => switchMode('VIEW')} className="submit-button"
              style={{ marginTop: '0.5rem', background: 'transparent', border: '1px solid var(--color-text-muted)' }}
            > Cancel·lar
            </button>
          </  div>
        )}

        {error && <p className="error-message" style={{ color: 'red' }}>{error}</p>}

      </div>
    </div>
  );
}
export default RepCount;