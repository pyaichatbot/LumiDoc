import logging
import uuid
from ai.metadata_store import MetadataStore
import faiss
import numpy as np
import os
import pickle
import asyncio


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
# Initialize stores
md_store = MetadataStore()

class VectorFAISS:
    def __init__(self, embedding_dim=1024):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = {}  # Store document ID -> vector mapping
        self.load_index()  # Load existing FAISS index if available

    def load_index(self):
        """Load FAISS index from file."""
        if os.path.exists(FAISS_INDEX_PATH):
            with open(FAISS_INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)

    def save_index(self):
        """Save FAISS index to file."""
        with open(FAISS_INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
  

    def search_vector(self, query_vector: np.ndarray, top_k=5):
        """Perform vector similarity search with normalized scoring."""
        try:
            distances, indices = self.index.search(np.array([query_vector]).astype(np.float32), top_k)
            max_distance = max(distances[0]) or 1
            results = {list(self.metadata.keys())[i]: 1 - (dist / max_distance) for i, dist in enumerate(distances[0])}
            return sorted(results.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logging.error(f"[FAISS] Error in vector search: {str(e)}")
            return []
        
    def store_vectors(self, filename, content, embedding_model, chunk_size=500):
        """
        Stores document chunks as vector embeddings in FAISS.
        
        Args:
            filename (str): Document filename.
            content (str): Full document text.
            embedding_model: Preloaded embedding model.
            chunk_size (int): Size of each text chunk.
        """
        try:
            chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
            embeddings = embedding_model.encode(chunks, convert_to_numpy=True)


            faiss_indices = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                faiss_index = self.index.ntotal  # Current FAISS index
                self.index.add(np.array([embedding]))  # Add to FAISS
                faiss_indices.append(faiss_index)
                self.metadata[len(self.index) - 1] = {"document_id": filename, "chunk": chunk}

            logging.info(f"[FAISS] Stored {len(chunks)} vector embeddings for {filename}")
            return faiss_indices

        except Exception as e:
            logging.error(f"[FAISS] Error storing vectors for {filename}: {str(e)}")
    
    async def search_async(self, query: str, embedding_model, top_k: int = 3, chat_document_id: str = None):
        """Performs Truly async FAISS vector search (Supports Open-Ended Queries)"""
        query_vector = embedding_model.encode(query, convert_to_numpy=True).reshape(1, -1)
        
        if query_vector.shape[1] != self.index.d:
            logging.error(f"[FAISS] Dimension mismatch: Query {query_vector.shape[1]} != Index {self.index.d}")
            return []
    
        # ✅ Run FAISS search in a non-blocking thread
        distances, indices = await asyncio.to_thread(self.index.search, query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Ignore invalid results
                doc_metadata = self.metadata.get(idx, {})
                #doc_metadata = await md_store.get_document_metadata_by_faiss_index(idx)  # Fetch from DB
                results.append({
                    "document_id": doc_metadata.get("document_id", chat_document_id),
                    "score": 1 / (1 + dist),  # Convert L2 distance to similarity score
                    "content": doc_metadata.get("content")
                })
        return results