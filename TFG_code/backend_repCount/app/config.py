import os
import sys

MAX_AGE_MINUTES = 60
ACCESS_TOKEN_EXPIRE_MINUTES = 30

TEMP_UPLOAD_PATH = '/datatmp2/joan/tfg_joan/temp_web/uploads'
TEMP_OUTPUT_PATH = '/datatmp2/joan/tfg_joan/temp_web/outputs'
os.makedirs(TEMP_UPLOAD_PATH, exist_ok=True)
os.makedirs(TEMP_OUTPUT_PATH, exist_ok=True)

SQLALCHEMY_DATABASE_URL = "sqlite:///./TFG_code/backend_repCount/data/usuaris.db"

CHUNK_SIZE = 1024 * 1024  # 1MB