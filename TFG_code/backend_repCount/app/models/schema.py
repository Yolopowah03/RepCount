from pydantic import BaseModel, EmailStr # type: ignore
from typing import Optional
from datetime import datetime

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
    token_type: Optional[str] = None
    access_token: Optional[str] = None
    class Config:
        from_attributes = True
        
class HistoryResponse(BaseModel):
    timestamp: datetime
    exercise: str
    rep_count: int
    video_path: str
    class Config:
        from_attributes = True
    thumbnail_name: Optional[str] = None
        
class VideoProcessResponse(BaseModel):
    video_name: str
    image_name: str
    count: int
    predicted_exercise: str
    message: Optional[str] = None
    
class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    
class UserModRequest (BaseModel):
    og_username: str
    og_email: str
    og_full_name: str
    full_name: str
    username: str
    email: EmailStr