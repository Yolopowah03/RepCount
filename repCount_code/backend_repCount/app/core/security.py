from passlib.context import CryptContext # type: ignore
from jose import jwt # type: ignore
from fastapi.security import OAuth2PasswordBearer # type: ignore
from fastapi import HTTPException # type: ignore
from datetime import datetime, timedelta, timezone
from typing import Optional
import re

SECRET_KEY = "hc0J27zAL07F4j8qSQFQwEyNv4IK3dyo"
ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def validate_password_strength(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim 8 caràcters.")
    if len(password) > 30:
        raise HTTPException(status_code=400, detail="La contrasenya no pot tenir més de 30 caràcters.")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim un número.")
    if not re.search(r"[a-zA-Z]", password):
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim una lletra.")
    
    return True