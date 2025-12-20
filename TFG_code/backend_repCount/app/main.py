import os
import shutil
import time
from fastapi import FastAPI, HTTPException, status, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from sqlalchemy import create_engine # type: ignore
from sqlalchemy.ext.declarative import declarative_base # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore
from apscheduler.schedulers.background import BackgroundScheduler # type: ignore
from .config import TEMP_OUTPUT_PATH, TEMP_UPLOAD_PATH # type: ignore
from .routers import users # type: ignore
from .routers import processing # type: ignore
from .models.user_model import Base, engine # type: ignore
from .utils import clean_files # type: ignore


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080","http://localhost:8079"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=TEMP_OUTPUT_PATH), name="outputs")

app.include_router(users.router)
app.include_router(processing.router)

@app.on_event("startup")
def start_scheduler():
    scheduler.add_job(clean_files, 'interval', minutes=180)
    scheduler.start()
    
@app.on_event("shutdown")
def shutdown_scheduler():
    # Detiene el scheduler cuando el servidor se apaga
    clean_files()
    scheduler.shutdown()
    
scheduler = BackgroundScheduler()