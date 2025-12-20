import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface UserData {
    username: string;
    password: string;
}

const Login = (): React.ReactElement => {
    const navigate = useNavigate();

    const [userData, setUserData] = useState<UserData>({
        username: '',
        password: '',
    });

    const [error, setError] = useState<string>('');
    const [success, setSuccess] = useState<boolean>(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setUserData({
            ...userData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setError('');
        setSuccess(false);

        try {
            const formData = new URLSearchParams();
            formData.append('username', userData.username);
            formData.append('password', userData.password);

            const response = await fetch('http://localhost:8080/users/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: formData
            });

            const data = await response.json();
 
            if (!response.ok) {
                const errorMessage = data.detail || 'Error al iniciar sessió.';
                setError(errorMessage);
                throw new Error(errorMessage);
            }

            console.log('Login successful:', data);

            localStorage.setItem('access_token', data.access_token);
            localStorage.setItem('username', data.username);
            localStorage.setItem('full_name', data.full_name);
            localStorage.setItem('email', data.email);

            const expirationTime = Date.now() + (30 * 60 * 1000); 
            localStorage.setItem('token_expiry', expirationTime.toString());

            setSuccess(true);

            setTimeout(() => navigate('/'), 2000);

        } catch (error: any) {
            setError(error.message); 
        }

    };

    return (
        <div className="card-container">
            <button onClick={() => navigate('/')} className="back-button">
                ← Pàgina principal
            </button>

            <h1 className="gradient-title">Iniciar sessió</h1>

            {success ? (
                <div className="success-message"><p> Sessió iniciada amb èxit! </p></div>
            ) : (
                <form onSubmit={handleSubmit} className="login-form">
                    <div className="form-group">
                    <label>Nom d'Usuari o Email:</label>
                    <input
                    type="text"
                    name="username"
                    value={userData.username}
                    onChange={handleChange}
                    required
                    className="text-input"
                    />
                    </div>

                    <div className="form-group">
                        <label>Contrasenya:</label>
                        <input
                        type="password"
                        name="password"
                        value={userData.password}
                        onChange={handleChange}
                        required
                        className="text-input"
                        />
                    </div>

                    <div className="register-prompt">
                        <p> No tens compte? </p>
                        <button  className="register-button"
                            onClick={() => navigate('/register')}>
                            Crear un compte nou
                        </button>
                    </div>

                        {error && <p className="error-message" style={{color: 'red'}}>{error}</p>}
                        <button type="submit" className="submit-button">Iniciar sessió</button>
                </form>
            )}

        </div>
    );
};


export default Login;