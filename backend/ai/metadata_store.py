import math
import psycopg2
import os
import logging
from sqlalchemy import select, text
import numpy as np
import datetime
import math
from concurrent.futures import ThreadPoolExecutor
import asyncio
from langdetect import detect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# PostgreSQL Configuration
POSTGRES_DB = os.getenv("POSTGRES_DB", "lumidoc_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lumidoc_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "lumidoc_pass")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
# Database URL with psycopg driver
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

async_engine = create_async_engine(DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


class MetadataStore:
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_URL)
        self.cursor = self.conn.cursor()
        self.executor = ThreadPoolExecutor(max_workers=5)  # Optimize parallel execution
        self.create_table()

    def create_table(self):
        """Creates the metadata table with additional tracking for chat-based interactions."""
        try:
            # Add this line at the beginning of the method
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    document_id TEXT PRIMARY KEY,
                    chat_id TEXT,
                    upload_time TIMESTAMP,
                    file_size INT,
                    file_type TEXT,
                    language TEXT,
                    processing_status TEXT,
                    vectorized BOOLEAN,
                    content TEXT,
                    retrieval_count INT DEFAULT 0,
                    likes INT DEFAULT 0,
                    dislikes INT DEFAULT 0,
                    faiss_index SERIAL,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS content_idx ON file_metadata USING gin(to_tsvector('english', content));")

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id SERIAL PRIMARY KEY,
                    document_id TEXT,
                    chunk_index INT,
                    chunk_content TEXT,
                    embedding vector(1024)
                )
            """)
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);")
            #CREATE EXTENSION IF NOT EXISTS vector; # todo in prod: run this in the database

            self.create_interaction_log_table();

            self.conn.commit()
            logging.info("[PostgreSQL] Metadata table initialized with full-text search and chat tracking")
        except Exception as e:
            logging.error(f"[PostgreSQL] Error creating metadata table: {str(e)}")

    def store_metadata(self, document_id, metadata, chat_id: str = None):
        """Inserts or updates metadata, including retrieval tracking fields."""
        try:
            self.cursor.execute("""
                INSERT INTO file_metadata (document_id, chat_id, upload_time, file_size, file_type, language, processing_status, vectorized, content, retrieval_count, likes, dislikes, last_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 0, 0, 0, CURRENT_TIMESTAMP)
                ON CONFLICT (document_id) DO UPDATE SET
                    upload_time = EXCLUDED.upload_time,
                    file_size = EXCLUDED.file_size,
                    file_type = EXCLUDED.file_type,
                    language = EXCLUDED.language,
                    processing_status = EXCLUDED.processing_status,
                    vectorized = EXCLUDED.vectorized,
                    content = EXCLUDED.content,
                    last_used = CURRENT_TIMESTAMP
            """, (document_id, chat_id, metadata["upload_time"], metadata["file_size"], metadata["file_type"], metadata["language"], metadata["processing_status"], metadata["vectorized"], metadata["content"]))
            self.conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] Error inserting metadata: {str(e)}")

    def keyword_search(self, query, top_k=5):
        """Full-text search in metadata."""
        try:
            self.cursor.execute("""
                SELECT document_id FROM file_metadata
                WHERE to_tsvector('english', content) @@ to_tsquery(%s)
                LIMIT %s
            """, (query.replace(" ", " & "), top_k))
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except Exception as e:
            logging.error(f"❌ Error in keyword search: {str(e)}")
            return []
        
    def record_document_usage(self, document_id):
        """Increments retrieval count when a document is used in chat."""
        try:
            self.cursor.execute("""
                UPDATE file_metadata
                SET retrieval_count = retrieval_count + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE document_id = %s
            """, (document_id,))
            self.conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] Error updating document usage: {str(e)}")

    def get_document_usage_score(self, document_id):
        """Retrieves document usage score to influence ranking."""
        self.cursor.execute("""
            SELECT retrieval_count FROM file_metadata WHERE document_id = %s
        """, (document_id,))
        result = self.cursor.fetchone()
        return result[0] if result else 0

    def update_user_feedback(self, document_id, feedback_type):
        """Updates like/dislike counts based on user feedback."""
        column = "likes" if feedback_type == "like" else "dislikes"
        try:
            self.cursor.execute(f"""
                UPDATE file_metadata
                SET {column} = {column} + 1
                WHERE document_id = %s
            """, (document_id,))
            self.conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] Error updating {feedback_type} feedback: {str(e)}")


    def get_time_decay_weight(self, document_id, decay_factor=0.05):
        """Computes time-decay weight for ranking."""
        self.cursor.execute("""
            SELECT last_used FROM file_metadata WHERE document_id = %s
        """, (document_id,))
        result = self.cursor.fetchone()

        if not result:
            return 1.0  # Default score if no data exists

        last_used = result[0]
        days_since_last_used = (document_id.datetime.utcnow() - last_used).days

        # Apply exponential decay formula: e^(-decay_factor * days)
        return math.exp(-decay_factor * days_since_last_used)


    def create_interaction_log_table(self):
        """Creates a table to store user interactions for training ML model."""
        try:
            # First create the table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_interactions (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    retrieval_count INT DEFAULT 0,
                    likes INT DEFAULT 0,
                    dislikes INT DEFAULT 0,
                    time_decay FLOAT DEFAULT 1.0,
                    relevance_score FLOAT DEFAULT 0.0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Then add the document_id column if it doesn't exist
            self.cursor.execute("""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name = 'search_interactions' 
                        AND column_name = 'document_id'
                    ) THEN
                        ALTER TABLE search_interactions 
                        ADD COLUMN document_id TEXT NOT NULL DEFAULT 'AI_Generated_Response';
                    END IF;
                END $$;
            """)
            
            self.conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] Error creating interaction log table: {str(e)}")
            self.conn.rollback()

    def log_search_interaction(self, query, filename, retrieval_count, likes, dislikes, time_decay, relevance_score):
        """Logs search interactions for ML-based ranking optimization."""
        try:
            # First, rollback any failed transaction
            self.conn.rollback()
            
            # Truncate query if it's too long (PostgreSQL has limits)
            max_query_length = 1000  # Adjust this value based on your needs
            if len(query) > max_query_length:
                query = query[:max_query_length] + "..."
            
            self.cursor.execute("""
                INSERT INTO search_interactions (document_id, query, retrieval_count, likes, dislikes, time_decay, relevance_score, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (filename, query, retrieval_count, likes, dislikes, time_decay, relevance_score))
            self.conn.commit()
        except Exception as e:
            logging.error(f"[PostgreSQL] Error logging search interaction: {str(e)}")
            self.conn.rollback()  # Rollback on error
            

    async def store_document_chunks(self, filename, content, embedding_model, chunk_size=500):
        """Splits a document into chunks and stores their embeddings in parallel."""
        try:
            chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
            logging.info(f"Created {len(chunks)} chunks for {filename}")
            
            embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
            logging.info(f"Generated embeddings with shape {embeddings.shape}")

            async with AsyncSessionLocal() as session:
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    assert embedding.shape[0] == 1024, f"Embedding dimension mismatch! Expected 1024, got {embedding.shape[0]}"
                    await session.execute(
                        text("""
                            INSERT INTO document_chunks (document_id, chunk_index, chunk_content, embedding)
                            VALUES (:filename, :chunk_index, :chunk_content, :embedding)
                            ON CONFLICT (document_id, chunk_index) DO NOTHING
                        """),
                        {
                            "filename": filename,
                            "chunk_index": i,
                            "chunk_content": chunk,
                            "embedding": embedding.tolist()
                        }
                    )
                await session.commit()
                
            logging.info(f"Successfully stored {len(chunks)} document chunks for {filename}")
            return chunks

        except Exception as e:
            logging.error(f"Error processing document {filename}: {str(e)}")
            raise

    async def get_top_chunks(self, query_vector, top_k=3, file_type=None, date_range=None):
        """Retrieve the most relevant document chunks with optional metadata filtering."""
        try:
            async with AsyncSessionLocal() as session:
                base_query = """
                    SELECT chunk_content, embedding 
                    FROM document_chunks 
                    JOIN file_metadata ON document_chunks.document_id = file_metadata.document_id 
                    WHERE 1=1
                """
                params = {}

                if file_type:
                    base_query += " AND file_metadata.file_type = :file_type"
                    params["file_type"] = file_type
                
                if date_range:
                    base_query += " AND file_metadata.upload_time >= NOW() - INTERVAL :date_range DAY"
                    params["date_range"] = date_range

                result = await session.execute(text(base_query), params)
                all_chunks = result.fetchall()

                if not all_chunks:
                    return []

                # Compute similarity scores using vectorized operations
                chunk_embeddings = np.array([np.array(chunk[1]) for chunk in all_chunks])
                assert chunk_embeddings.shape[1] == 1024, "Embedding dimension mismatch in DB!"
                similarity_scores = np.dot(chunk_embeddings, query_vector) / (np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_vector))
                sorted_indices = np.argsort(similarity_scores)[-top_k:][::-1]

                return [all_chunks[i][0] for i in sorted_indices]  # Return top chunks
        except Exception as e:
            logging.error(f"[PostgreSQL] Error retrieving top chunks with filters: {str(e)}")
            return []
        
    async def keyword_search_async(self, query: str, top_k: int = 3, chat_document_id: str = None):
        """Perform async metadata keyword search in PostgreSQL with multilingual support."""        
        # ✅ Detect language dynamically
        detected_lang = detect(query)  # Detects "en", "de", "fr", etc.

        # ✅ Map to PostgreSQL full-text search configurations
        lang_mapping = {
            "en": "english",
            "de": "german",
            "fr": "french",
            "es": "spanish",
            "it": "italian",
        }
        
        pg_lang = lang_mapping.get(detected_lang, "english")  # Default to English
       
        session = await self.get_async_session()
        try:
            result = await session.execute(text("""
                SELECT document_id, content FROM file_metadata
                WHERE to_tsvector(:pg_lang, content) @@ to_tsquery(:query)
                LIMIT :top_k
            """), {
                "pg_lang": pg_lang,  # Use detected language
                "query": query.replace(" ", " & "),
                "top_k": top_k
            })
            rows = result.fetchall()
            return [{"document_id": row[0], "content": row[1]} for row in rows]
        finally:
            await session.close()
            
    def get_document_metadata(self, doc_index):
        """Retrieve document metadata based on FAISS index"""
        self.cursor.execute(
            "SELECT document_id, content FROM file_metadata WHERE faiss_index = %s",
            (doc_index,)
        )
        result = self.cursor.fetchone()
        if result:
            return {"document_id": result[0], "content": result[1]}
        return {"document_id": None, "content": "No content found"}
    
    async def get_async_session(self) -> AsyncSession:
        """Returns a new async session for database operations."""
        return AsyncSessionLocal()

    def get_likes(self, document_id: str) -> int:
        """Retrieve the number of likes for a document."""
        try:
            self.cursor.execute("""
                SELECT likes FROM file_metadata WHERE document_id = %s
            """, (document_id,))
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            logging.error(f"[PostgreSQL] Error fetching likes for {document_id}: {str(e)}")
            return 0

    def get_dislikes(self, document_id: str) -> int:
        """Retrieve the number of dislikes for a document."""
        try:
            self.cursor.execute("""
                SELECT dislikes FROM file_metadata WHERE document_id = %s
            """, (document_id,))
            result = self.cursor.fetchone()
            return result[0] if result else 0
        except Exception as e:
            logging.error(f"[PostgreSQL] Error fetching dislikes for {document_id}: {str(e)}")
            return 0
        
    async def get_files_for_chat(self, chat_id: str):
        """Retrieves metadata of files uploaded within a chat session."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    text("""
                        SELECT document_id, file_type, upload_time 
                        FROM file_metadata 
                        WHERE chat_id = :chat_id
                    """),
                    {"chat_id": chat_id}
                )
                rows = result.fetchall()
                return [{"document_id": row[0], "file_type": row[1], "upload_time": row[2]} for row in rows]
        except Exception as e:
            logging.error(f"[PostgreSQL] Error retrieving files for chat {chat_id}: {str(e)}")
            return []