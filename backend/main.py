import json
import traceback
import uuid
from ai.metadata_store import MetadataStore
from ai.vector_store import VectorFAISS
from ai.elasticsearch_store import ElasticsearchStore
from fastapi import FastAPI, Depends, status, HTTPException, Request, Form
from fastapi import File, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from database import Base, check_database_connection, engine, get_chat_session, get_db, execute_db_operation, save_chat_session
from sqlalchemy.orm import Session
from sqlalchemy import text
from models import ChatSession, ChatSessionRequest, ChatSessionResponse, UserCreate, Token, UserResponse, User, TokenData, UserChatSession
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import List, Optional
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
import logging
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel
from redis_cache import cache_chat_session, delete_cached_chat_session, get_cached_chat_session, update_cached_chat_session
from ai.ml_ranking import RankingModel
import asyncio
from ai.llm_response_generator import LLMResponseGenerator
from sentence_transformers import SentenceTransformer
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
import docx  # python-docx for DOCX text extraction
import pandas as pd  # Pandas for Excel text extraction
import pytesseract  # Tesseract OCR for image-based text extraction
import re  # Add missing import for sanitize_filename

from file_process import FileStorage


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize stores
metadata_store = MetadataStore()
vector_store = VectorFAISS()
elasticsearch_store = ElasticsearchStore()
ranking_model = RankingModel()
llm_generator = LLMResponseGenerator()
file_storage = FileStorage()
try:
    os.environ["HF_HOME"] = "/tmp/huggingface"
    #BAAI/bge-m3
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-large", cache_folder="/tmp/huggingface")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    embedding_model = SentenceTransformer("BAAI/bge-small-en")  # ✅ Fallback model 

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(
    title="LumiDoc API",
    description="API for LumiDoc application",
    version="1.0.0"
)

# Add rate limiting error handler
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middlewares
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Directory to temporarily store uploaded files (Mock - Replace with actual S3 or Database storage)
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Authentication helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_tokens(user_id: str) -> tuple[str, str, datetime, datetime]:
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token_expires_at = datetime.utcnow() + access_token_expires
    refresh_token_expires_at = datetime.utcnow() + refresh_token_expires
    
    access_token = create_token(
        data={"sub": user_id, "type": "access"},
        expires_delta=access_token_expires
    )
    refresh_token = create_token(
        data={"sub": user_id, "type": "refresh"},
        expires_delta=refresh_token_expires
    )
    
    return access_token, refresh_token, access_token_expires_at, refresh_token_expires_at

async def get_current_user(
    request: Request,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_type: str = payload.get("type")
        exp: datetime = datetime.fromtimestamp(payload.get("exp"))
        
        if email is None or token_type != "access" or exp < datetime.utcnow():
            raise credentials_exception
            
        token_data = TokenData(email=email, exp=exp)
    except JWTError:
        raise credentials_exception
    
    # Get user and validate session
    user = db.query(User).filter(User.email == token_data.email).first()
    if not user:
        raise credentials_exception
        
    # Update session last activity
    session = db.query(UserChatSession).filter(
        UserChatSession.user_id == user.id,
        UserChatSession.access_token == token,
        UserChatSession.is_active == True
    ).first()
    
    if not session:
        raise credentials_exception
        
    session.last_activity = datetime.utcnow()
    db.commit()
    
    return user

@app.on_event("startup")
async def startup_event():
    """Initialize database tables and start periodic ML training."""
    try:
        metadata_store.create_interaction_log_table()  # ✅ Ensure table exists
        logging.info("✅ Interaction Log Table Initialized.")
    except Exception as e:
        logging.error(f"❌ Failed to create interaction log table: {str(e)}")
        return

    try:
        #asyncio.create_task(ranking_model.periodic_training())  # ✅ Start ML ranking model updates
        background_tasks = BackgroundTasks()
        background_tasks.add_task(ranking_model.periodic_training)
        logging.info("✅ Periodic ML Training Started.")
    except Exception as e:
        logging.error(f"❌ Failed to start periodic training: {str(e)}")

@app.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["authentication"])
@limiter.limit("5/minute")
async def signup(request: Request, user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Check for existing user
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = User(
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name,
            is_active=True
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"✅ User {user.email} registered successfully")  # Debugging log
        return UserResponse(
            id=new_user.id,
            full_name=new_user.full_name,
            email=new_user.email,
            is_active=new_user.is_active,
            created_at=new_user.created_at.strftime("%Y-%m-%d %H:%M:%S")
        )         
        #return UserResponse.model_validate(new_user)
    except Exception as e:
        logger.error(f"Signup failed: {str(e)}")
        db.rollback()
        print(f"❌ ERROR in /signup: {str(e)}")  # Print error to FastAPI logs
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@app.delete("/delete_user", tags=["authentication"], status_code=status.HTTP_200_OK)
async def delete_user(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete user account."""
    try:
        # Delete all chat sessions for the user
        db.query(UserChatSession).filter(UserChatSession.user_id == user.id).delete()
        # Delete all chat sessions for the user
        db.query(ChatSession).filter(ChatSession.user_id == user.id).delete()
        # Delete the user account
        db.query(User).filter(User.email == user.email).delete()
        db.commit()
        return {"message": "User account deleted successfully"}
    except Exception as e:
        logger.error(f"Delete user failed: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/login", response_model=Token, tags=["authentication"])
@limiter.limit("10/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 compatible token login."""
    try:
        print(f"🔑 Login attempt for user: {form_data.username}")  # Debugging log
        # Authenticate user
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token, refresh_token, access_expires, refresh_expires = create_tokens(user.email)
        
        # Create session
        session_id = str(uuid.uuid4())
        new_session = UserChatSession(
            session_id=session_id,
            user_id=user.id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=refresh_expires,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host,
            is_active=True
        )
        
        # Deactivate other sessions for this user
        db.query(UserChatSession).filter(
            UserChatSession.user_id == user.id,
            UserChatSession.is_active == True
        ).update({"is_active": False})
        
        db.add(new_session)
        db.commit()
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            session_id=session_id,
            expires_at=access_expires,
            refresh_token=refresh_token,
            user_id=user.id
        )
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/refresh", response_model=Token, tags=["authentication"])
async def refresh_token(
    request: Request,
    refresh_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    try:
        # Validate refresh token
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            token_type = payload.get("type")
            
            if not email or token_type != "refresh":
                raise HTTPException(
                    status_code=401, 
                    detail="Invalid refresh token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token format",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Get session
        session = db.query(UserChatSession).filter(
            UserChatSession.refresh_token == refresh_token,
            UserChatSession.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired session",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create new tokens
        access_token, new_refresh_token, access_expires, refresh_expires = create_tokens(email)
        
        # Update session
        session.access_token = access_token
        session.refresh_token = new_refresh_token
        session.expires_at = refresh_expires
        session.last_activity = datetime.utcnow()
        
        db.commit()
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            session_id=session.session_id,
            expires_at=access_expires,
            refresh_token=new_refresh_token,
            user_id=session.user_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during token refresh",
            headers={"WWW-Authenticate": "Bearer"}
        )

@app.post("/logout", tags=["authentication"])
async def logout(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Logout user and invalidate current session."""
    try:
        token = request.headers.get("authorization").split(" ")[1]
        session = db.query(UserChatSession).filter(
            UserChatSession.access_token == token,
            UserChatSession.is_active == True
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
        
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Logout failed")

@app.get("/me", response_model=UserResponse, tags=["authentication"])
async def get_user_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Get information about the currently authenticated user.
    Returns:
        UserResponse: The current user's information
    """
    user = db.query(User).filter(User.email == current_user.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(email=user.email, created_at=user.created_at.strftime("%Y-%m-%d %H:%M:%S"))

@app.get("/sessions", tags=["authentication"])
async def get_user_sessions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all active sessions for current user."""
    sessions = db.query(UserChatSession).filter(
        UserChatSession.user_id == user.id,
        UserChatSession.is_active == True
    ).all()
    return sessions

@app.delete("/sessions/{session_id}", tags=["authentication"])
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a specific session."""
    session = db.query(UserChatSession).filter(
        UserChatSession.session_id == session_id,
        UserChatSession.user_id == user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.is_active = False
    db.commit()
    return {"message": "Session terminated successfully"}

@app.get("/chat_sessions/{user_id}/{chat_id}", tags=["chat"])
async def get_user_chat_sessions(
    user_id: int,  # Change type hint to int
    chat_id: str,
    db: Session = Depends(get_db)
):
    """Get all active sessions for current user."""    
    try:
        chat_sessions = db.query(ChatSession).filter(
            ChatSession.chat_id == chat_id,
            ChatSession.user_id == user_id,  # No need for type casting
            ChatSession.is_active == True
        ).all()
        return chat_sessions
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")

@app.delete("/chat_sessions/{user_id}/{chat_id}", tags=["chat"])
async def delete_chat_session(
    user_id: int,  # Change to int type
    chat_id: str,
    db: Session = Depends(get_db)
):
    """Delete a specific chat session."""
    try:
        chat_session = db.query(ChatSession).filter(
            ChatSession.chat_id == chat_id,
            ChatSession.user_id == user_id,
            ChatSession.is_active == True
        ).first()
        
        if not chat_session:
            raise HTTPException(status_code=404, detail="Chat Session not found")
        
        chat_session.is_active = False
        db.delete(chat_session)
        db.commit()
        return {"message": "Session terminated successfully"}
    except Exception as e:
        logging.error(f"Error deleting chat session: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error occurred: {str(e)}")

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint to verify service status, including database connectivity.
    """
    try:
        db_status = "connected" if check_database_connection() else "not connected"
        status = "healthy" if db_status == "connected" else "unhealthy"
        return {"status": status, "database": db_status}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "database": "not connected", "error": str(e)}
    
async def retrieve_context_async(query: str, top_k: int = 3, file_type: str = None, date_range: int = None, chat_id: str = None):
    """Parallel Hybrid Retrieval with ML Ranking"""

     # ✅ Encode query for FAISS search
    query_vector = embedding_model.encode([query], convert_to_numpy=True)[0]
    # Ensure correct dimensionality
    expected_dim = 1024  # Adjust this based on your model
    if query_vector.shape[0] != expected_dim:
        logging.error(f"Query vector has incorrect dimensionality: {query_vector.shape[0]} != {expected_dim}")
        print(f"Query vector has incorrect dimensionality: {query_vector.shape[0]} != {expected_dim}")
        query_vector = query_vector[:expected_dim]  # Truncate if too long
        query_vector = np.pad(query_vector, (0, max(0, expected_dim - query_vector.shape[0])))

    chat_document_id = f"chat_{uuid.uuid4().hex}"

    # ✅ Run all searches asynchronously
    vector_task = asyncio.create_task(vector_store.search_async(query, embedding_model, top_k, chat_document_id))
    keyword_task = asyncio.create_task(elasticsearch_store.search_documents_async(query, top_k, chat_document_id))
    metadata_task = asyncio.create_task(metadata_store.keyword_search_async(query, top_k))
    chunk_task = asyncio.create_task(metadata_store.get_top_chunks(query_vector, top_k, file_type, date_range))  # ✅ Get relevant document chunks

    # ✅ Get files uploaded in the chat session
    file_task = asyncio.create_task(metadata_store.get_files_for_chat(chat_id)) if chat_id else None

    # ✅ Wait for all tasks to complete
    search_tasks = [vector_task, keyword_task, metadata_task, chunk_task]
    if file_task:
        search_tasks.append(file_task)

    search_results = await asyncio.gather(*search_tasks)

    # ✅ Ensure results are lists and not coroutines
    faiss_results, es_results, metadata_results, chunk_results = search_results[:4]
    file_results = search_results[4] if file_task else []

    faiss_results = faiss_results if isinstance(faiss_results, list) else []
    es_results = es_results if isinstance(es_results, list) else []
    #metadata_results = metadata_results if isinstance(metadata_results, list) else []
    chunk_results = chunk_results if isinstance(chunk_results, list) else []
    file_results = file_results if isinstance(file_results, list) else []
    # ✅ Combine and Deduplicate Results
    combined_results = {}

    def add_result(result, source, score_key, score_value):
        """Merge results and assign scores dynamically."""
        doc_id = result.get("document_id", chat_document_id)  # Generate ID for open-ended chat
        if doc_id not in combined_results:
            combined_results[doc_id] = {
                "document_id": doc_id,
                "faiss_score": 0,
                "es_score": 0,
                "metadata_score": 0,
                "chunk_score": 0,
                "retrieval_count": metadata_store.get_document_usage_score(doc_id),
                "last_used": metadata_store.get_last_used_time(doc_id),
                "content": result["content"],
                "likes": metadata_store.get_likes(doc_id),
                "dislikes": metadata_store.get_dislikes(doc_id),
                "time_decay": ranking_model.compute_decay(metadata_store.get_last_used_time(doc_id)),
            }
        combined_results[doc_id][score_key] = score_value

    # ✅ Populate results
    for result in faiss_results:
        add_result(result, "faiss", "faiss_score", result["score"])
    for result in es_results:
        add_result(result, "es", "es_score", result["es_score"])
    """ for result in metadata_results:
        add_result(result, "metadata", "metadata_score", 1.0) """
    for result in chunk_results:
        add_result(result, "chunk_score", 1.5)  # ✅ Higher weight for chunk similarity
    for result in file_results:
        add_result(result, "file", "file_score", 1.0)

    # ✅ Convert dictionary to list
    results_list = list(combined_results.values())

    # ✅ Compute initial scores before ML ranking
    for doc in results_list:
        doc["initial_score"] = ranking_model.score_document(doc)

    # ✅ Apply ML-Based Ranking
    ranked_results = ranking_model.rank_results(results_list)
    
    if not ranked_results:
        # ✅ Log AI-only interaction
        metadata_store.log_search_interaction(
            query=query,
            filename="AI_Generated_Response",  # No document retrieved
            retrieval_count=0,
            likes=0,
            dislikes=0,
            time_decay=0,
            relevance_score=0
        )
    else:
        # ✅ Log Search Interaction (after ranking)
        for doc in ranked_results[:top_k]:
            metadata_store.log_search_interaction(
                query=query,
                filename=chat_document_id if doc["document_id"] == chat_document_id else doc["document_id"],
                retrieval_count=doc["retrieval_count"],
                likes=doc["likes"],
                dislikes=doc["dislikes"],
                time_decay=doc["time_decay"],
                relevance_score=doc["initial_score"]
        )

    return ranked_results[:top_k]

@app.post("/chat_response/", response_model=ChatSessionResponse)
@limiter.limit("30/minute")
async def chat_response(request: Request, db: Session = Depends(get_db)):
    """
    Handles user chat queries, retrieving context from Redis cache, hybrid retrieval, or LLM fallback.
    """
    try:
        data = await request.json()  # ✅ Extract JSON data manually
        chat_id = data.get("chat_id")
        user_id = data.get("user_id")
        query = data.get("query")   
        query = query.strip()
        #delete_chat_session(chat_id, user_id)    

        # ✅ Step 1: Check Redis Cache First
        cached_chat_response = get_cached_chat_session(chat_id, user_id)
        if cached_chat_response:
            print("✅ Returning chat from Redis cache")            

            # ✅ Step 2: Hybrid Retrieval (Metadata + FAISS + Elasticsearch)
            context = await retrieve_context_async(query, top_k=3, chat_id=chat_id)
            # context_text = "\n\n".join(context)

            # ✅ Step 3: Generate Chatbot Response Using LLM (Placeholder for now)
            # ✅ Step 3: If No Context Found → Use LLM Fallback
            if not context:
                print("⚠️ No relevant documents found. Generating LLM Response...")
                llm_response = await llm_generator.generate_response_without_context(query)
            else:
                llm_response = await llm_generator.generate_response(query, context)

            new_message = {"query": query, "response": llm_response}
            
            update_cached_chat_session(chat_id, user_id, new_message)  # ✅ Update Redis cache     
            chat_session = save_chat_session(db, chat_id, user_id, new_message)
            # ✅ Convert messages explicitly to JSON serializable format
            messages_serialized = json.loads(json.dumps(chat_session.messages))
            return ChatSessionResponse(
                chat_id=chat_session.chat_id,
                user_id=chat_session.user_id,
                messages=messages_serialized,
                created_at=chat_session.created_at.strftime("%Y-%m-%d %H:%M:%S") if chat_session.created_at else None,
                updated_at=chat_session.updated_at.strftime("%Y-%m-%d %H:%M:%S") if chat_session.updated_at else None,
                title=chat_session.title
            )

       # ✅ Step 2: Hybrid Retrieval (Metadata + FAISS + Elasticsearch)
        context = await retrieve_context_async(query, top_k=3, chat_id=chat_id)
        #context_text = "\n\n".join(retrieved_contexts)

        # ✅ Step 3: Generate Chatbot Response Using LLM (Placeholder for now)
        # ✅ Step 3: If No Context Found → Use LLM Fallback
        if not context:
            print("⚠️ No relevant documents found. Generating LLM Response...")
            llm_response = await llm_generator.generate_response_without_context(query)
        else:
            llm_response = await llm_generator.generate_response(query, context)

        # ✅ Step 4: Store or Update the Chat Session
        chat_session = get_chat_session(db, chat_id, user_id)

        if not chat_id or not user_id or not query:
            raise HTTPException(status_code=400, detail="Missing required fields (chat_id, user_id, query)")
        
        chat_session = save_chat_session(db, chat_id, user_id, {"query": query, "response": llm_response})
                      
        # ✅ Convert messages explicitly to JSON serializable format
        messages_serialized = json.loads(json.dumps(chat_session.messages))

        response = ChatSessionResponse(
            chat_id=chat_session.chat_id,
            user_id=chat_session.user_id,
            messages=messages_serialized,
            created_at=chat_session.created_at.strftime("%Y-%m-%d %H:%M:%S") if chat_session.created_at else None,
            updated_at=chat_session.updated_at.strftime("%Y-%m-%d %H:%M:%S") if chat_session.updated_at else None,
            title=chat_session.title
        )
        cache_chat_session(chat_id, user_id, response)  # ✅ Cache chat session in Redis
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Function to sanitize filenames
def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename

@app.post("/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    chat_id: str = Query(..., description="Chat ID to associate uploaded files with"),
    files: List[UploadFile] = File(..., description="Files to upload")
):
    """Upload multiple files and process them asynchronously."""
    uploaded_files = []
    
    for file in files:
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            # Validate file type
            content_type = file.content_type or "application/octet-stream"
            
            # Sanitize filename
            safe_filename = sanitize_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, safe_filename)
            
            # Read file content
            contents = await file.read()
            
            # Validate file size (e.g., max 10MB)
            if len(contents) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds size limit of 10MB.")
               
            metadata = {
                "document_id": file.filename,
                "chat_id": chat_id,
                "upload_time": datetime.utcnow().isoformat(),
                "file_size": len(contents),
                "file_type": content_type,
                "language": "",
                "processing_status": "pending",
                "vectorized": False,
                "content": None  # Will be populated during processing
            }
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(contents)
            
            uploaded_files.append(file.filename)
            
            # Add to background tasks
            background_tasks.add_task(process_file, file_path, file.filename, metadata)
            
        except Exception as e:
            logging.error(f"Error saving file {file.filename}: {str(e)} - Trace: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}. Error: {str(e)}")
        finally:
            await file.close()  # Ensure file is closed
    
    return {"status": "success", "uploaded_files": uploaded_files}

def process_file(file_path: str, document_id: str, metadata: dict):
    try:
        logging.info(f"Processing file: {document_id}")
        extracted_text = extract_text(file_path, document_id)
        if not extracted_text:
            logging.warning(f"Empty content extracted from {document_id}")
            return
        
        # ✅ Split into chunks
        chunk_size = 500
        content_chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size)]
        embeddings = embedding_model.encode(content_chunks, convert_to_numpy=True)  # Encode in batch

        # ✅ Store document chunks in PostgreSQL & ✅ Store vector embeddings in FAISS
        # Change this line to use asyncio.run since we're in a synchronous context
        asyncio.run(process_and_store_document(document_id, extracted_text, metadata.get("chat_id"), embedding_model))

        # ✅ Store document in Elasticsearch for full-text search
        elasticsearch_store.store_document(document_id, metadata["file_type"], content_chunks, embeddings)
        
        # Ensure the embedding has the correct dimension (512 or 768)
        #if embeddings.shape[0] not in [512, 768]:
        #    logging.warning(f"Unexpected embedding dimension: {embeddings.shape[0]}. Adjusting to 512.")
        #    embeddings = np.pad(embeddings, (0, max(0, 512 - embeddings.shape[0])))
 
        # ✅ Store metadata in PostgreSQL
        metadata["embedding"] = embeddings.tolist()
        metadata["vectorized"] = True
        metadata_store.store_metadata(document_id, metadata)
        
        logging.info(f"Successfully processed and embedded file: {document_id}")
    except Exception as e:
        logging.error(f"Error processing file {document_id}: {str(e)} - Trace: {traceback.format_exc()}")

def extract_text(file_path: str, document_id: str) -> str:
    """Extracts text based on file type."""
    
    try:
        if document_id.lower().endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif document_id.lower().endswith(".docx"):
            return extract_text_from_docx(file_path)
        elif document_id.lower().endswith(".xlsx"):
            return extract_text_from_xlsx(file_path)
        elif document_id.lower().endswith((".png", ".jpg", ".jpeg")):
            return extract_text_from_image(file_path)
        else:
            logging.warning(f"Unsupported file format: {document_id}")
            return ""

    except Exception as e:
        logging.error(f"Error extracting text from {document_id}: {str(e)}")
        return ""
    
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        logging.error(f"Error extracting text from PDF {file_path}: {str(e)} - Trace: {traceback.format_exc()}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error extracting text from DOCX {file_path}: {str(e)} - Trace: {traceback.format_exc()}")
    return text


def extract_text_from_xlsx(file_path: str) -> str:
    text = ""
    try:
        df = pd.read_excel(file_path)
        text = df.to_string()
    except Exception as e:
        logging.error(f"Error extracting text from XLSX {file_path}: {str(e)} - Trace: {traceback.format_exc()}")
    return text


def extract_text_from_image(file_path: str) -> str:
    text = ""
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"Error extracting text from image {file_path}: {str(e)} - Trace: {traceback.format_exc()}")
    return text

async def process_and_store_document(filename: str, content: str, chat_id: str, embedding_model):
    """Stores document chunks in PostgreSQL and vectors in FAISS asynchronously."""
    try:
        # Store in PostgreSQL (Metadata)
        chunks = await metadata_store.store_document_chunks(filename, content, embedding_model)

        if chunks:
            # Store in FAISS (Vector Search)
            faiss_indices = vector_store.store_vectors(filename, content, embedding_model)

            # Update metadata to link FAISS indices
            async with metadata_store.get_async_session() as session:
                for i, faiss_id in enumerate(faiss_indices):
                    await session.execute(
                        text("""
                            UPDATE document_chunks 
                            SET faiss_index = :faiss_id
                            WHERE filename = :filename AND chunk_index = :chunk_index
                        """),
                        {
                            "faiss_id": faiss_id,
                            "filename": filename,
                            "chunk_index": i
                        }
                    )
                await session.commit()
    except Exception as e:
        logging.error(f"Error in process_and_store_document: {str(e)}")
        raise
