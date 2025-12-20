from sqlalchemy.orm import declarative_base, relationship # type: ignore
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime # type: ignore
from datetime import datetime, timezone
from fastapi.security import OAuth2PasswordBearer # type: ignore
from ..config import SQLALCHEMY_DATABASE_URL # type: ignore
from sqlalchemy import create_engine # type: ignore
from sqlalchemy.orm import sessionmaker # type: ignore

Base = declarative_base()

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
    thumbnail_name = Column(String, nullable=True, default=None)
    
    owner = relationship("UserDB", back_populates="history")