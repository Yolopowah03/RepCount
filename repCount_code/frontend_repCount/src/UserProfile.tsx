import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './User.css';

type ViewMode = 'VIEW' | 'EDIT' | 'PASSWORD' | 'DELETE' | 'CLOSING_SESSION';

const UserProfile = (): React.ReactElement => {
    const navigate = useNavigate();

    const [viewMode, setViewMode] = useState<ViewMode>('VIEW');

    const [userData, setUserData] = useState({
        username: '',
        fullName: '',
        email: ''
    });

    const [passwordData, setPasswordData] = useState({
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
    });

    const [error, setError] = useState<string>('');
    const [success, setSuccess] = useState<boolean>(false);
    const [successMessage, setSuccessMessage] = useState<string>('');

    useEffect(() => {

        const token = localStorage.getItem('access_token');

        if (!token) {
            navigate('/login');

        } else {

            setUserData({
                username: localStorage.getItem('username') || '',
                fullName: localStorage.getItem('full_name') || '',
                email: localStorage.getItem('email') || ''
            })
        }
    }, [navigate]);

    useEffect(() => {
        const expiryString = localStorage.getItem('token_expiry');
        const token = localStorage.getItem('access_token');

        if (!token || !expiryString) {
            setUserData({ username: '', fullName: '', email: '' });
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

    }, [userData, navigate]);

    const forceLogOut = () => {
        localStorage.clear();
        setUserData({ username: '', fullName: '', email: '' });
        setError('La sessió ha caducat. Si us plau, inicia sessió de nou.');
    }

    const switchMode = (mode: ViewMode) => {
        setViewMode(mode);
        setError('');
        setSuccessMessage('');
        setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
    };

    const handleLogOut = () => {
        localStorage.clear();
        navigate('/');
    };

    const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setUserData({
            ...userData,
            [e.target.name]: e.target.value
        });

        localStorage.setItem('username', userData.username);
        localStorage.setItem('full_name', userData.fullName);
        localStorage.setItem('email', userData.email);
    };

    const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setPasswordData({
            ...passwordData,
            [e.target.name]: e.target.value
        });
    };

    const handleSaveProfile = async () => {
        setError('');
        try {
            const token = localStorage.getItem('access_token');

            const response = await fetch('http://172.23.192.1:8080/users/mod_profile/', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    og_username: localStorage.getItem('username') || '',
                    og_email: localStorage.getItem('email') || '',
                    og_full_name: localStorage.getItem('full_name') || '',
                    full_name: userData.fullName,
                    username: userData.username,
                    email: userData.email
                })
            });
            const data = await response.json();
            if (!response.ok) {
                const errorMessage = data.detail || 'Error al iniciar sessió.';
                setError(errorMessage);
                throw new Error(errorMessage);
            }

            setSuccessMessage('Perfil actualitzat correctament');
            setTimeout(() => switchMode('VIEW'), 2000);
        } catch (error: any) {
            setError(error.message);
        }
    };

    const handleChangePassword = async () => {
        setError('');

        if (passwordData.newPassword !== passwordData.confirmPassword) {
            setError('Les noves contrasenyes no coincideixen.');
            return;
        }

        try {
            const token = localStorage.getItem('access_token');

            const response = await fetch('http://your_port1:8080/users/change_password/', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    current_password: passwordData.currentPassword,
                    new_password: passwordData.newPassword
                })
            });

            if (!response.ok) {
                const data = await response.json();
                const errorMessage = data.detail || 'Error al canviar la contrasenya.';
                setError(errorMessage);
                throw new Error(errorMessage);
            }

            setSuccessMessage('Contrasenya canviada correctament');
            setTimeout(() => switchMode('VIEW'), 2000);

        } catch (error: any) {
            setError(error.message);
        }
    };

    const handleDeleteAccount = async () => {
        setError('');
        try {
            const token = localStorage.getItem('access_token');

            const response = await fetch('http://172.23.192.1:8080/users/delete_account/', {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                const data = await response.json();
                const errorMessage = data.detail || 'Error al esborrar el compte.';
                setError(errorMessage);
                throw new Error(errorMessage);
            }

            localStorage.clear();

            setSuccessMessage('Compte esborrat correctament');
            setSuccess(true);
            setTimeout(() => switchMode('VIEW'), 2000);
        }
        catch (error: any) {
            setError(error.message);
        }
    };

    const HandleBack = () => {
        if (viewMode === 'VIEW') {
            navigate('/');
        } else {
            switchMode('VIEW');
        }
    };

    return (
        <div className="card-container">
            <button onClick={HandleBack} className="back-button">
                ← {viewMode === 'VIEW' ? 'Pàgina Principal' : 'Cancel·lar'}
            </button>

            <h1 className="gradient-title">
                {viewMode === 'VIEW' && 'El Meu Perfil'}
                {viewMode === 'EDIT' && 'Editar Perfil'}
                {viewMode === 'PASSWORD' && 'Canviar Contrasenya'}
                {viewMode === 'DELETE' && 'Esborrar Compte'}
            </h1>

            <div className="profile-display">

                {viewMode === 'VIEW' && (
                    <>

                        {successMessage && <div className="success-message" style={{ marginBottom: '1rem' }}>{successMessage}</div>}

                        <div className="form-group">
                            <label htmlFor="full_name">Nom complet:</label>
                            <p className="text-display">
                                {userData.fullName}
                            </p>
                        </div>
                        <div className="form-group">
                            <label>Nom d'Usuari:</label>
                            <p className="text-display"                        >
                                {userData.username}
                            </p>
                        </div>

                        <div className="form-group">
                            <label>Email:</label>
                            <p className="text-display">
                                {userData.email}
                            </p>
                        </div>

                        <button className="submit-button" onClick={() => switchMode('EDIT')}>
                            Editar Perfil
                        </button>

                        <button className="submit-button" onClick={() => switchMode('PASSWORD')}>
                            Canviar Contrasenya
                        </button>

                        <button className="submit-button logout-button" onClick={() => switchMode('CLOSING_SESSION')}>
                            Tancar Sessió
                        </button>

                        <button className="submit-button logout-button" onClick={() => switchMode('DELETE')}>
                            Esborrar Compte
                        </button>
                    </>
                )}

                {viewMode === 'EDIT' && (
                    <>
                        <div className="form-group">
                            <label>Nom complet:</label>
                            <input
                                type="text"
                                name="fullName"
                                value={userData.fullName}
                                onChange={handleProfileChange}
                                className="text-input"
                            />
                        </div>
                        <div className="form-group">
                            <label>Nom d'Usuari:</label>
                            <input
                                type="text"
                                name="username"
                                value={userData.username}
                                onChange={handleProfileChange}
                                className="text-input"
                            />
                        </div>
                        <div className="form-group">
                            <label>Email:</label>
                            <input
                                type="email"
                                name="email"
                                value={userData.email}
                                onChange={handleProfileChange}
                                className="text-input"
                            />
                        </div>

                        {error && <p className="error-message">{error}</p>}
                        {successMessage && <div className="success-message">{successMessage}</div>}

                        <button className="submit-button" onClick={handleSaveProfile}>
                            Guardar Canvis
                        </button>
                    </>
                )}

                {viewMode === 'PASSWORD' && (
                    <>
                        <div className="form-group">
                            <label>Contrasenya Actual:</label>
                            <input
                                type="password"
                                name="currentPassword"
                                value={passwordData.currentPassword}
                                onChange={handlePasswordChange}
                                className="text-input"
                            />
                        </div>
                        <div className="form-group">
                            <label>Nova Contrasenya:</label>
                            <input
                                type="password"
                                name="newPassword"
                                value={passwordData.newPassword}
                                onChange={handlePasswordChange}
                                className="text-input"
                            />
                        </div>
                        <div className="form-group">
                            <label>Confirma la Nova Contrasenya:</label>
                            <input
                                type="password"
                                name="confirmPassword"
                                value={passwordData.confirmPassword}
                                onChange={handlePasswordChange}
                                className="text-input"
                            />
                        </div>

                        <button className="submit-button" onClick={handleChangePassword}>
                            Canviar Contrasenya
                        </button>

                        {error && <p className="error-message">{error}</p>}
                        {successMessage && <div className="success-message">{successMessage}</div>}
                    </>
                )}

                {viewMode === 'DELETE' && (
                    success ? (
                        <div className="success-message delete-account">Compte esborrat correctament. </div>
                    ) : (
                        <div style={{ textAlign: 'center' }}>
                            <p>Estàs segur que vols esborrar el teu compte? Aquesta acció és irreversible.</p>

                            {error && <p className="error-message">{error}</p>}


                            <button className="submit-button logout-button" onClick={handleDeleteAccount}>
                                Esborrar Compte
                            </button>

                            <button onClick={() => switchMode('VIEW')} className="submit-button"
                                style={{ marginTop: '0.5rem', background: 'transparent', border: '1px solid var(--color-text-muted)' }}
                            > Cancel·lar
                            </button>
                        </div>
                    )
                )}

                {viewMode === 'CLOSING_SESSION' && (
                    <div style={{ textAlign: 'center' }}>
                        <p>Vols tancar sessió?</p>

                        {error && <p className="error-message">{error}</p>}

                        <button className="submit-button logout-button" onClick={handleLogOut}>
                            Tancar Sessió
                        </button>

                        <button onClick={() => switchMode('VIEW')} className="submit-button"
                            style={{ marginTop: '0.5rem', background: 'transparent', border: '1px solid var(--color-text-muted)' }}
                        > Cancel·lar
                        </button>
                    </div>
                )}

            </div>
        </div>
    );

};


export default UserProfile;