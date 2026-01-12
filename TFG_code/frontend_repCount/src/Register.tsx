import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './User.css'; 

interface UserData {
    username: string;
    email: string;
    password: string;
    full_name: string;
}

const Register = (): React.ReactElement => {
    const navigate = useNavigate();

    const [userData, setUserData] = useState<UserData>({
        username: '',
        email: '',
        password: '',
        full_name: ''
    });

    const [error, setError] = useState<string>('');
    const [success, setSuccess] = useState<boolean>(false);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setUserData({
            ...userData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError('');
        setSuccess(false);

        try {
            const response = await fetch('http://localhost:8080/users/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(userData)
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMessage = data.detail || 'Error al crear el compte.';
                setError(errorMessage);
                throw new Error(errorMessage);
            }

            setSuccess(true);

            setTimeout(() => navigate('/login'), 2000);

        } catch (error: any) {
            setError(error.message);
        }
    }; 

    return (
        <div className="card-container">

            {success ? (<span className="back-button-spacer"></span>)
                    : (<button onClick={() => navigate('/')} className="back-button">
                        ← Pàgina principal
                    </button>)
            }
            
            <h1 className="gradient-title">Crear compte</h1>

            {success ? (
                <div className="success-message"><p> Compte creat amb èxit! </p>
                <button 
                    onClick={() => navigate('/login')}
                    className="submit-button">
                    Iniciar sessió
                </button>
                </div>

            ) : (

                <form onSubmit={handleSubmit} className="register-form">
                    <div className="form-group">
                        <label htmlFor="full_name">Nom complet:</label>
                        <input
                            type='text'
                            id='full_name'
                            name='full_name'
                            value={userData.full_name}
                            onChange={handleChange}
                            required
                            className="text-input"
                        />
                    </div>
                    <div className="form-group">
                    <label>Nom d'Usuari:</label>
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
                    <label>Email:</label>
                    <input
                    type="email"
                    name="email"
                    value={userData.email}
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
                    <p> Ja tens compte? </p>
                    <button 
                        onClick={() => navigate('/login')}
                        className="register-button">
                        Iniciar sessió
                    </button>
                </div>

                {error && <p className="error-message" style={{color: 'red'}}>{error}</p>}
                <button type="submit" className="submit-button">Registrar-se</button>
                </form>

            )}
        </div>
    );
};


export default Register;