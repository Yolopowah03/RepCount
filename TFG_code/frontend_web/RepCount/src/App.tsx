import React, { useState, useEffect } from 'react';

function App(): React.ReactElement {

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processedVideoUrl, setProcessedVideoUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>('');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

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


  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setMessage('');
      setProcessedVideoUrl(null);
    } else {
      setSelectedFile(null);
      setMessage('Selecciona un arxiu de vídeo vàlid.');
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setMessage('Selecciona un vídeo per pujar.');
      return;
    }

    setIsLoading(true);
    setMessage('Pujant i processant el vídeo...');
    setProcessedVideoUrl(null);

    // --- SIMULACIÓ DE LA CRIDA AL BACKEND ---
    setTimeout(() => {
      if (previewUrl) setProcessedVideoUrl(previewUrl);
      setIsLoading(false);
      setMessage('¡Video procesado exitosamente!');
    }, 3000);

  };

  return (
  
  <div className="card-container">
    <h1 className="gradient-title">
      RepCount
    </h1>
    <p className="description-text">
      Puja un video, i et comptarem les repeticions d'exercicis que has fet!
    </p>

    <form onSubmit={handleSubmit} className="form-space">
      <div className="form-group">
        <label htmlFor="video-upload">
          Selecciona el teu video:
        </label>
        <input
          id="video-upload"
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="file-input"
        />
      </div>
      <button type="submit"
      disabled={isLoading || !selectedFile}
      className="submit-button">
        {isLoading ? (
          <>
            Processant...
          </>
        ) : (
          'Comptar Repeticions'
        )}
      </button>
    </form>

    {message && (
      <p className="message-text">
        {message}
      </p>
    )}

    <div className="video-results">
      {previewUrl && (
        <div className="video-box">
          <h2 className="original-title">Vista Previa (Original)</h2>
          <video 
            key={previewUrl}
            width="100%"
            controls
          >
            <source src={previewUrl} type={selectedFile?.type} />
            El teu navegador no soporta el format del video.
          </video>
        </div>
      )}

      {processedVideoUrl && (
              <div className="video-box">
                <h2 className="processed-title">Video Procesado</h2>
                <video
                  key={processedVideoUrl}
                  width="100%"
                  controls
                  autoPlay
                >
                  <source src={processedVideoUrl} type={selectedFile?.type} />
                  Tu navegador no soporta la etiqueta de video.
                </video>
              </div>
      )}
    </div>
  </div>

  );
}

export default App;