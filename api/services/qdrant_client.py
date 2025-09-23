from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import numpy as np
from typing import List, Optional
from api.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedding_size = settings.EMBEDDING_SIZE
        self.connect()
        self.ensure_collection()

    def connect(self):
        """Connect to Qdrant database"""
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=60  # Increased timeout for operations
            )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def upsert_embeddings(self, vectors: List[dict]):
        """Insert or update embeddings in the database"""
        try:
            points = [
                models.PointStruct(
                    id=vec["id"],
                    vector=vec["vector"].tolist() if hasattr(vec["vector"], 'tolist') else vec["vector"],
                    payload=vec["payload"]
                )
                for vec in vectors
            ]
            
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} vectors: {operation_info}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert embeddings: {e}")
            return False

    def search_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        """Search for similar vectors"""
        try:
            if hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

# Create a global instance
qdrant_service = QdrantService()