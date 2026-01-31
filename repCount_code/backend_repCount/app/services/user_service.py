from ..models.user_model import UserDB, SessionLocal, oauth2_scheme # type: ignore
from sqlalchemy.orm import Session # type: ignore
from fastapi import Depends, HTTPException, status, Query # type: ignore
from jose import JWTError, jwt # type: ignore
from ..core.security import SECRET_KEY, ALGORITHM # type: ignore
from ..config import SQLALCHEMY_DATABASE_URL # type: ignore



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No s'ha pogut validar l'usuari i/o la contrasenya.",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_user_url(token: str = Query(..., alias="token"), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No s'ha pogut validar l'usuari i/o la contrasenya.",
        headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user