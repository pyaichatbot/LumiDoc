from datetime import datetime
import os
from sqlalchemy import create_engine, Column, String, JSON, TIMESTAMP, func, text
from sqlalchemy.orm import sessionmaker, Session
from db_base import Base  # Import from the new file
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from typing import Any, Dict, Generator
import logging
from fastapi import HTTPException
from models import User, ChatSession
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# PostgreSQL Configuration
POSTGRES_DB = os.getenv("POSTGRES_DB", "lumidoc_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lumidoc_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lumidoc_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Database URL with psycopg driver
DATABASE_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Database Engine & Session
try:
    # Create database engine with optimized settings
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        pool_timeout=30,
        pool_pre_ping=True,
        max_overflow=10,
        echo=os.getenv("SQL_ECHO", "false").lower() == "true",
        connect_args={
            "connect_timeout": 10,
            "application_name": "LumiDoc API"
        }
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Error connecting to database: {e}")
    raise

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get a database session.
    Includes error handling and proper session cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Database error occurred"
        )
    finally:
        db.close()

def init_db() -> None:
    """
    Initialize database tables with error handling.
    Should be called when application starts.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        print("✅ Database tables initialized successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        print(f"❌ Error initializing database: {e}")
        raise

""" def check_db_connection() -> bool:
    
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False
    finally:
        db.close() """

# Health check function
def check_database_connection() -> bool:
    """
    Check if database connection is working.
    Returns True if connection is successful, False otherwise.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

# Database operation with error handling
def execute_db_operation(operation, *args, **kwargs):
    """
    Generic function to execute database operations with error handling.
    """
    db = SessionLocal()
    try:
        result = operation(db, *args, **kwargs)
        db.commit()
        return result
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database operation failed: {str(e)}")
        raise
    finally:
        db.close()

# Chat Session Management
def save_chat_session(db: Session, chat_id: str, user_id: str, message: Dict[str, Any]):
    chat_session = db.query(ChatSession).filter(ChatSession.chat_id == chat_id, ChatSession.user_id == user_id).first()
    if chat_session:
        chat_session.messages.append(message)
        chat_session.updated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    else:
        chat_session = ChatSession(chat_id=chat_id, user_id=user_id, messages=[message], title=message["query"][:15])
        db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session

def get_chat_session(db: Session, chat_id: str, user_id: str):
    return db.query(ChatSession).filter(ChatSession.chat_id == chat_id, ChatSession.user_id == user_id).first()

def delete_chat_session(db: Session, chat_id: str):
    chat_session = db.query(ChatSession).filter(ChatSession.chat_id == chat_id).first()
    if chat_session:
        db.delete(chat_session)
        db.commit()
        return True
    return False
