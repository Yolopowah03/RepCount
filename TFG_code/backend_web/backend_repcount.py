
# Imports
import shutil
import os
import sys
from typing import Optional
import time
from datetime import datetime, timedelta, timezone
import re

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, status, Depends # type: ignore
from fastapi.responses import FileResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # type: ignore

from pydantic import BaseModel, EmailStr # type: ignore
from apscheduler.schedulers.background import BackgroundScheduler # type: ignore

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker, Session, relationship # type: ignore
from passlib.context import CryptContext # type: ignore
from jose import JWTError, jwt # type: ignore

PYTHON_REPCOUNT_PATH = '/datatmp2/joan/tfg_joan/TFG_code/repCount'
sys.path.append(PYTHON_REPCOUNT_PATH)

MAX_AGE_MINUTES = 60

TEMP_UPLOAD_PATH = '/datatmp2/joan/tfg_joan/temp_web/uploads'
TEMP_OUTPUT_PATH = '/datatmp2/joan/tfg_joan/temp_web/outputs'
os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)
os.makedirs(TEMP_OUTPUT_PATH, exist_ok=True)

try:
    from repCount_YOLO11_web import repcount_main # type: ignore
except ImportError:
    raise NotImplementedError("Error d'importació del mòdul repCount_YOLO11_web.")

# Configuració base de dades usuaris

SECRET_KEY = "hc0J27zAL07F4j8qSQFQwEyNv4IK3dyo"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SQLALCHEMY_DATABASE_URL = "sqlite:///./usuaris.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

#Configuració seguretat

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

#Clases i funcions usuaris

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    history = relationship("HistoryDB", back_populates="owner")
    
class HistoryDB(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise = Column(String)
    rep_count = Column(Integer)
    video_path = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    owner = relationship("UserDB", back_populates="history")
    
Base.metadata.create_all(bind=engine)
    
class UserCreate(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    password: str
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class UserResponse(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    class Config:
        orm_mode = True
        
class HistoryResponse(BaseModel):
    timestamp: datetime
    exercise: str
    rep_count: int
    video_path: str
    class Config:
        orm_mode = True
        
class VideoProcessResponse(BaseModel):
    video_url: str
    image_url: str
    count: int
    predicted_exercise: str
    message: Optional[str] = None
    
# Funcions auxiliars base de dades
    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
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

def validate_password_strength(password: str):
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim 8 caràcters.")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim un número.")
    if not re.search(r"[a-zA-Z]", password):
        raise HTTPException(status_code=400, detail="La contrasenya ha de tenir com a mínim una lletra.")
    
    return True

# Neteja d'arxius

scheduler = BackgroundScheduler()

def clean_files():
    
    print("Netejant arxius...")
    
    dirs_to_clean = [TEMP_UPLOAD_PATH, TEMP_OUTPUT_PATH]
    
    for directory in dirs_to_clean:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error al eliminar {filename}: {e}")
                    
# App set up
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=TEMP_OUTPUT_PATH), name="outputs")

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(clean_files, 'interval', minutes=15)
    scheduler.start()

@app.on_event("shutdown")
def shutdown_scheduler():
    # Detiene el scheduler cuando el servidor se apaga
    clean_files()
    scheduler.shutdown()
    
# User DB endpoints

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    validate_password_strength(user.password)
    
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

@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No s'ha pogut validar l'usuari i/o la contrasenya.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

@app.get("/users/me/history", response_model=list[HistoryResponse])
def get_user_history(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    return current_user.history

# App event endpoins
        
@app.post(
    "/repcount",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_201_CREATED
)

def repcount_endpoint(
    file: UploadFile = File(...),
    skip_frames: int = Form(1, ge=1, description="Saltar frames per a processament més ràpid (2 = mantenir 1 de cada 2, 3 = mantenir 1 de cada 3...)"),
    vel_reduction: float = Form(1.0, ge=0, description="Reduïr velocitat de vídeo de sortida"),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    if not file.filename.endswith(('.mp4')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Introdueix un arxiu de vídeo amb format .mp4")
    
    #Si fps_reduction en format 0.5, 0.25 convertir a 2, 4...
    if vel_reduction < 1:
        vel_reduction = 1 / vel_reduction
    
    timestamp = int(time.time())
    
    file_name = file.filename.split(".")[0] if "." in file.filename else "video"
    file_name = "".join(x for x in file_name if x.isalnum())
    extension = file.filename.split(".")[-1] if "." in file.filename else "mp4"
    
    input_file_name = f"{file_name}_{timestamp}_input.{extension}"
    output_video_name = f"{current_user.username}_{file_name}_{timestamp}_out.mp4"
    output_image_name = f"{current_user.username}_{file_name}_{timestamp}_out.jpg"
    
    input_filepath = os.path.join(TEMP_UPLOAD_PATH, input_file_name)
    output_video_path = os.path.join(TEMP_OUTPUT_PATH, output_video_name) # Video
    output_image_path = os.path.join(TEMP_OUTPUT_PATH, output_image_name)
    
    try:
        with open(input_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        args = {
            'input_video_path': input_filepath,
            'output_path_video': output_video_path,
            'output_path_img': output_image_path,
            'output_dir': None,
            'skip_frames': skip_frames,
            'fps_reduction': vel_reduction
        }
        
        count, pred_label = repcount_main(args)
        
        history_entry = HistoryDB(
            user_id=current_user.id,
            exercise=pred_label,
            rep_count=count,
            video_path=output_video_path
        )
        db.add(history_entry)
        db.commit()
        
        web_video_url=f"/outputs/{output_video_name}"
        web_image_url=f"/outputs/{output_image_name}"

        return VideoProcessResponse(
            video_url=web_video_url,
            image_url=web_image_url,
            count=count,
            predicted_exercise=pred_label
        )
        
    
    except Exception as e:
        # 500 Internal Server Error
        clean_files()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al processament del vídeo: {str(e)}"
        )
        
@app.get("/download/{file_name}/{file_type}")
def download_file(file_name: str, file_type: str, current_user: UserDB = Depends(get_current_user)):
    
    if not file_name.startswith(current_user.username):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tens permís per accedir a aquest fitxer.")
        
    if file_type == 'video':
        file_path = os.path.join(TEMP_OUTPUT_PATH, f"{file_name}_out.mp4")
        mediatype = 'video/mp4'
    elif file_type == 'image':
        file_path = os.path.join(TEMP_OUTPUT_PATH, f"{file_name}_out.jpg")
        mediatype = 'image/jpg'
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tipus de fitxer invàlid. Utilitza un vídeo o una imatge.")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fitxer no trobat.")
        
    return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type=mediatype)