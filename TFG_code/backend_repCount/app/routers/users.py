from fastapi import APIRouter, Depends, HTTPException, status # type: ignore
from sqlalchemy.orm import sessionmaker, Session # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from ..models.user_model import UserDB
from ..models.schema import UserCreate, UserResponse, HistoryResponse, PasswordChangeRequest, UserModRequest # type: ignore
from ..core.security import get_password_hash, verify_password, create_access_token, validate_password_strength
from ..services.user_service import get_current_user, get_db # type: ignore
from fastapi.security import OAuth2PasswordRequestForm # type: ignore
from datetime import timedelta
from ..config import ACCESS_TOKEN_EXPIRE_MINUTES # type: ignore

router = APIRouter(
    tags=["users"],
    prefix="/users" # Opcional: Define un prefijo común para todas las rutas
)

Base = declarative_base()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    validate_password_strength(user.password)
    
    if len(user.username) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom d'usuari no pot tenir més de 30 caràcters.")
    if len(user.full_name) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom complet no pot tenir més de 30 caràcters.")
    if len(user.email) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="L'email no pot tenir més de 30 caràcters.")
    
    if db.query(UserDB).filter(UserDB.username == user.username).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom d'usuari ja existeix.")
    if db.query(UserDB).filter(UserDB.email == user.email).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="L'email ja està registrat.")
    
    hashed_password = get_password_hash(user.password)
    db_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login", response_model=UserResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    
    user = db.query(UserDB).filter((UserDB.username == form_data.username) | (UserDB.email == form_data.username)).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No s'ha trobat l'email / nom d'usuari. Prova a registrar el teu compte!",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La contrasenya introduïda no és correcta.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, 
            "token_type": "bearer", 
            "username": user.username,
            "full_name": user.full_name,
            "email": user.email
           }

@router.put("/change_password", status_code=status.HTTP_200_OK)
def change_password(password_change: PasswordChangeRequest, current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(password_change.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La contrasenya actual no és correcta.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    validate_password_strength(password_change.new_password)
    
    current_user.hashed_password = get_password_hash(password_change.new_password)
    db.add(current_user)
    db.commit()
    return


@router.put("/mod_profile", status_code=status.HTTP_200_OK)
def modify_profile(user_mod: UserModRequest, db: Session = Depends(get_db)):
    
    if user_mod.username == user_mod.og_username and user_mod.email == user_mod.og_email and user_mod.full_name == user_mod.og_full_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No s'ha realitzat cap canvi en el perfil.")
    
    if len(user_mod.username) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom d'usuari no pot tenir més de 30 caràcters.")
    if len(user_mod.full_name) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom complet no pot tenir més de 30 caràcters.")
    if len(user_mod.email) > 30:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="L'email no pot tenir més de 30 caràcters.")
    
    if user_mod.username != user_mod.og_username:
        if db.query(UserDB).filter(UserDB.username == user_mod.username).first():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El nom d'usuari ja existeix.")
    if user_mod.email != user_mod.og_email:
        if db.query(UserDB).filter(UserDB.email == user_mod.email).first():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="L'email ja està registrat.")
    
    db.query(UserDB).filter(UserDB.username == user_mod.og_username).update({
        UserDB.username: user_mod.username,
        UserDB.full_name: user_mod.full_name,
        UserDB.email: user_mod.email
    })
    db.commit()
    return {"message": "Perfil actualitzat correctament."}
    
@router.delete("/delete_account",  status_code=status.HTTP_200_OK)
def delete_account(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.commit()
    return

@router.get("/show_history", status_code=status.HTTP_200_OK, response_model=list[HistoryResponse])
def show_history(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    
    history_entries = db.query(UserDB).filter(UserDB.id == current_user.id).first().history
    return history_entries