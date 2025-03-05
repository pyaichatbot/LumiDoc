from sqlalchemy import JSON, TIMESTAMP, Column, Integer, String, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db_base import Base  # Import from the new file
from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# SQLAlchemy Models
class UserChatSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    last_activity = Column(TIMESTAMP, onupdate=func.now())
    is_active = Column(Boolean, default=True)
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    # Indexes
    __table_args__ = (
        Index('idx_session_token', 'access_token'),
        Index('idx_session_user', 'user_id', 'is_active'),
    )

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    sessions = relationship("UserChatSession", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
    )

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, unique=True, index=True)  # Added unique constraint
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)  # Added ondelete CASCADE
    title = Column(String, nullable=True)
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    files = relationship("FileMetadata", back_populates="chat", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_chat_user', 'user_id', 'is_active'),
    )

# Pydantic Models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    session_id: str
    user_id: int
    expires_at: datetime
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    email: Optional[str] = None
    exp: Optional[datetime] = None

class ChatSessionCreate(BaseModel):
    title: Optional[str] = None

class ChatSessionResponse(BaseModel):
    chat_id: str
    user_id: int
    messages: List[Dict[str, Any]]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    title: Optional[str] = None

class ChatSessionRequest(BaseModel):
    chat_id: str
    user_id: int  # ✅ Ensure user_id is an INTEGER
    query: str

class FileMetadata(Base):
    __tablename__ = "file_metadata"

    document_id = Column(String, primary_key=True)
    chat_id = Column(String, ForeignKey("chat_sessions.chat_id", ondelete="CASCADE"), nullable=True)
    upload_time = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String, nullable=True)
    language = Column(String, nullable=True)
    processing_status = Column(String, default="pending")
    vectorized = Column(Boolean, default=False)
    retrieval_count = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    last_used = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    # Relationship
    chat = relationship("ChatSession", back_populates="files")