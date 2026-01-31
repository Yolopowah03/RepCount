from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, status, Request, WebSocket, WebSocketDisconnect # type: ignore
from fastapi.responses import StreamingResponse # type: ignore
from sqlalchemy.orm import Session # type: ignore
from ..models.user_model import UserDB, HistoryDB
from ..models.schema import VideoProcessResponse # type: ignore
from ..services.user_service import get_current_user, get_current_user_url, get_db # type: ignore
from ..services.download_service import stream_file  # type: ignore
import os
import time
import shutil
import sys
from ..config import TEMP_UPLOAD_PATH, TEMP_OUTPUT_PATH # type: ignore
from ..utils import clean_files # type: ignore
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_progress(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)

PYTHON_REPCOUNT_PATH = '/datatmp2/joan/repCount/repCount_code/repCount'
sys.path.append(PYTHON_REPCOUNT_PATH)

try:
    from repCount_YOLO11_web import repcount_main # type: ignore
except ImportError:
    raise NotImplementedError("Error d'importació del mòdul repCount_YOLO11_web.")

router = APIRouter(
    tags=["repCount"],
    prefix="/repCount"
)

manager = ConnectionManager()
executor = ThreadPoolExecutor()

@router.websocket("/ws/progress/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@router.post(
    "/video_processing",
    response_model=VideoProcessResponse,
    status_code=status.HTTP_201_CREATED
)

async def repcount_endpoint(
    video_file: UploadFile = File(...),
    skip_frames: int = Form(1, ge=1, description="Saltar frames per a processament més ràpid (2 = mantenir 1 de cada 2, 3 = mantenir 1 de cada 3...)"),
    vel_reduction: float = Form(1.0, ge=0, description="Reduïr velocitat de vídeo de sortida"),
    client_id: str = Form(..., description="ID únic del client per a la connexió WebSocket"),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
       
    main_loop = asyncio.get_running_loop()
    
    def progress_reporter(percent, status_msg):
        asyncio.run_coroutine_threadsafe(
            manager.send_progress(client_id, {"progress": percent, "status": status_msg}),
            main_loop
        )
        
    if not video_file.filename.endswith(('.mp4')):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Introdueix un arxiu de vídeo amb format .mp4")
    
    #Si fps_reduction en format 0.5, 0.25 convertir a 2, 4...
    if vel_reduction < 1:
        vel_reduction = 1 / vel_reduction
    
    timestamp = int(time.time())
    
    file_name = video_file.filename.split(".")[0] if "." in video_file.filename else "video"
    file_name = "".join(x for x in file_name if x.isalnum())
    extension = video_file.filename.split(".")[-1] if "." in video_file.filename else "mp4"
    
    input_file_name = f"{file_name}_{timestamp}_input.{extension}"
    output_video_name = f"{current_user.username}_{file_name}_{timestamp}_out.mp4"
    output_thumbnail_name = f"{current_user.username}_{file_name}_{timestamp}_thumbnail.jpg"
    output_image_name = f"{current_user.username}_{file_name}_{timestamp}_out.jpg"
    
    input_filepath = os.path.join(TEMP_UPLOAD_PATH, input_file_name)
    output_video_path = os.path.join(TEMP_OUTPUT_PATH, output_video_name)
    output_thumbnail_path = os.path.join(TEMP_OUTPUT_PATH, output_thumbnail_name)
    output_image_path = os.path.join(TEMP_OUTPUT_PATH, output_image_name)
    
    try:
        
        loop = asyncio.get_event_loop()
        
        with open(input_filepath, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
            
        args = {
            'input_video_path': input_filepath,
            'output_path_video': output_video_path,
            'output_path_thumbnail': output_thumbnail_path,
            'output_path_img': output_image_path,
            'output_dir': None,
            'skip_frames': skip_frames,
            'fps_reduction': vel_reduction,
            'progress_callback': progress_reporter
        }
        
        count, pred_label = await main_loop.run_in_executor(None, lambda: repcount_main(args))

        history_entry = HistoryDB(
            user_id=current_user.id,
            exercise=pred_label,
            rep_count=count,
            video_path=output_video_path,
            thumbnail_name=output_thumbnail_name
        )
        
        try:
            db.add(history_entry)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error al desar l'entrada a l'historial: {e}")

        return VideoProcessResponse(
            video_name=output_video_name,
            image_name=output_image_name,
            count=count,
            predicted_exercise=pred_label
        )
        
    
    except Exception as e:
        await manager.send_progress(client_id, {"progress": 0, "status": f"Error al processament del vídeo: {str(e)}"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al processament del vídeo: {str(e)}"
        )
        
@router.get("/download/button/{file_name}/{file_type}")
def download_file(file_name: str, file_type: str, request: Request, current_user: UserDB = Depends(get_current_user)):
    
    if not file_name.startswith(current_user.username):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tens permís per accedir a aquest fitxer.")
    
    file_path = os.path.join(TEMP_OUTPUT_PATH,  f"{file_name}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Fitxer no trobat.")
        
    if file_type == 'video':
        mediatype = 'video/mp4'    
    elif file_type == 'image':
        mediatype = 'image/jpg'
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tipus de fitxer invàlid. Utilitza un vídeo o una imatge.")
    
    range_header = request.headers.get('range')
    file_size = os.path.getsize(file_path)
    
    headers = {
        'Content-Disposition': f'attachment; filename="{file_name}"',
        'Accept-Ranges': 'bytes',
        'Content-Type': mediatype
    }
    
    if range_header:
        start = int(range_header.strip().lower().split('bytes=')[1].split('-')[0])
        end = file_size - 1
        
        length = end - start
        
        headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
        headers['Content-Length'] = str(length + 1)
        status_code = status.HTTP_206_PARTIAL_CONTENT
    else:
        headers['Content-Length'] = str(file_size)
        status_code = status.HTTP_200_OK
    
    return StreamingResponse(
        stream_file(file_path, range_header, mediatype),
        status_code=status_code,
        media_type=mediatype,
        headers=headers
    )
    
@router.get("/download/visualization/{file_name}/{file_type}")
def download_file(file_name: str, file_type: str, request: Request, current_user: UserDB = Depends(get_current_user_url)):
    
    if not file_name.startswith(current_user.username):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tens permís per accedir a aquest fitxer.")
    
    file_path = os.path.join(TEMP_OUTPUT_PATH,  f"{file_name}")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Fitxer no trobat.")
        
    if file_type == 'video':
        mediatype = 'video/mp4'    
    elif file_type == 'image':
        mediatype = 'image/jpg'
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Tipus de fitxer invàlid. Utilitza un vídeo o una imatge.")
    
    range_header = request.headers.get('range')
    file_size = os.path.getsize(file_path)
    
    headers = {
        'Content-Disposition': f'inline; filename="{file_name}"',
        'Accept-Ranges': 'bytes',
        'Content-Type': mediatype
    }
    
    if range_header:
    
        start = int(range_header.strip().lower().split('bytes=')[1].split('-')[0])

        range_match = range_header.strip().lower().split('bytes=')[1].split('-')
        start = int(range_match[0])
        
        if len(range_match) > 1 and range_match[1]:
            end = int(range_match[1])
        else:
            end = file_size - 1
            
        end = min(end, file_size - 1)
        
        length = end - start + 1

        headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
        headers['Content-Length'] = str(length)

        status_code = status.HTTP_206_PARTIAL_CONTENT
        
    else:
        headers['Content-Length'] = str(file_size)
        status_code = status.HTTP_200_OK
    
    return StreamingResponse(
        stream_file(file_path, range_header, mediatype),
        status_code=status_code,
        media_type=mediatype,
        headers=headers
    )
        