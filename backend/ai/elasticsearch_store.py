from datetime import datetime, time
import logging
from elasticsearch import Elasticsearch
import os
import asyncio

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))  # ✅ Convert to int
ELASTICSEARCH_SCHEME = os.getenv("ELASTICSEARCH_SCHEME", "http")  # ✅ Add scheme
INDEX_NAME = "lumidoc_documents"

class ElasticsearchStore:
    def __init__(self, index_name=INDEX_NAME):
        self.index_name = index_name
        # Initialize Elasticsearch client
        self.client = Elasticsearch([{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT, "scheme": ELASTICSEARCH_SCHEME}], timeout=30,  # ✅ Increase timeout in case of slow startup
            max_retries=5,  # ✅ Retry connection in case of failures
            retry_on_timeout=True) # ✅ Prevent timeout failures)
        # ✅ Retry connecting until Elasticsearch is up
        self.wait_for_elasticsearch()
        self.initialize_index()

    def wait_for_elasticsearch(self):
        """Wait until Elasticsearch is available before proceeding."""
        for _ in range(10):
            try:
                if self.client.ping():
                    print("✅ Elasticsearch is up and running!")
                    return
            except Exception:
                print("⏳ Waiting for Elasticsearch to start...")
                time.sleep(5)  # Wait before retrying

        raise RuntimeError("❌ Elasticsearch did not start. Check your configuration.")


    # Ensure index exists
    def initialize_index(self):
        """Create Elasticsearch index with keyword & metadata filtering."""
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body={
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "document_id": {"type": "keyword"},
                        "upload_time": {"type": "date"},
                        "language": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "chunk_text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": 768}  # Adjust dims for embedding size
                    }
                }
            })

    # Store document in Elasticsearch
    def store_document(self, document_id, file_type, content_chunks, embeddings):
        """
        Store document chunks with embeddings in Elasticsearch.
        
        Args:
            filename (str): Name of the file.
            file_type (str): Type of the file (e.g., PDF, DOCX).
            content_chunks (list): List of document chunks.
            embeddings (list): Corresponding list of chunk embeddings.
        """
        try:
            for chunk_index, (chunk, embedding) in enumerate(zip(content_chunks, embeddings)):
                self.client.index(
                    index=self.index_name,
                    document={
                        "document_id": document_id,
                        "file_type": file_type,
                        "upload_time": datetime.utcnow(),
                        "chunk_index": chunk_index,
                        "chunk_text": chunk,
                        "embedding": embedding.tolist(),
                    },
                )
            logging.info(f"[Elasticsearch] Stored {len(content_chunks)} document chunks for {document_id}")

        except Exception as e:
            logging.error(f"[Elasticsearch] Error storing document {document_id}: {str(e)}")

    def index_document(self, filename, content, metadata):
            """Index document content and metadata for full-text search."""
            self.client.index(index=self.index_name, id=filename, body={
                "content": content,
                "document_id": metadata.get("document_id", ""),
                "upload_time": metadata.get("upload_time", ""),
                "language": metadata.get("language", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "chunk_text": metadata.get("chunk_text", ""),
                "embedding": metadata
            })

    def keyword_search(self, query, top_k=5, file_type=None, date_range=None):
            """Perform keyword search with optional filters."""
            filter_conditions = []

            if file_type:
                filter_conditions.append({"term": {"file_type": file_type}})
            if date_range:
                filter_conditions.append({"range": {"upload_time": {"gte": f"now-{date_range}d/d"}}})

            query_body = {
                "query": {
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": filter_conditions
                    }
                },
                "size": top_k
            }

            response = self.client.search(index=self.index_name, body=query_body)
            return [hit["_id"] for hit in response["hits"]["hits"]]

    # Perform keyword search
    def search_documents(self, query, top_k=5):
        response = self.client.search(index=INDEX_NAME, body={
            "query": {
                "match": {"content": query}
            },
            "size": top_k
        })
        return [hit["_source"] for hit in response["hits"]["hits"]]
    
    async def search_documents_async(self, query: str, top_k: int = 3, chat_document_id: str = None):
        """Performs truly async Elasticsearch keyword search."""
        
        result = await asyncio.to_thread(
            self.client.search,
            index=self.index_name,
            body={
                "query": {"match": {"content": query}},
                "size": top_k
            }
        )

        return [
            {
                "document_id": hit["_id"] if hit["_id"] != chat_document_id else chat_document_id,
                "es_score": hit["_score"],
                "content": hit["_source"]["content"]
            }
            for hit in result["hits"]["hits"]
        ]