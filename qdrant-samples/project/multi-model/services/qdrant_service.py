from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from models.animal import Animal
from utils.logger import Logger

class QdrantService:
    """Handle Qdrant vector database operations"""
    
    def __init__(self, host: str = "192.168.110.18", port: int = 6333):
        self.client = QdrantClient(host, port=port)
        self.logger = Logger()

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            return self.client.collection_exists(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection with proper vector configuration"""
        try:
            if self.collection_exists(collection_name):
                self.logger.info(f"Collection '{collection_name}' already exists.")
                return True

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                    "image": VectorParams(size=768, distance=Distance.COSINE)
                }
            )
            self.logger.info(f"Collection '{collection_name}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False

    def create_point(self, point_id: int, image_embedding: np.ndarray, 
                    text_embedding: np.ndarray, animal: Animal) -> Optional[PointStruct]:
        """Create a point structure for Qdrant"""
        try:
            if image_embedding is None or text_embedding is None:
                self.logger.warning(f"Invalid embeddings for animal: {animal.name}")
                return None

            # Ensure embeddings are in the correct format
            image_vec = image_embedding.tolist() if hasattr(image_embedding, 'tolist') else image_embedding
            text_vec = text_embedding.tolist() if hasattr(text_embedding, 'tolist') else text_embedding

            point = PointStruct(
                id=point_id,
                vector={
                    "image": image_vec,
                    "text": text_vec
                },
                payload=animal.to_dict()
            )
            return point
        except Exception as e:
            self.logger.error(f"Error creating point for {animal.name}: {e}")
            return None

    def upsert_points(self, collection_name: str, points: List[PointStruct]) -> bool:
        """Insert or update points in collection"""
        try:
            if not points:
                self.logger.warning("No points to upsert")
                return False

            # Filter out None points
            valid_points = [p for p in points if p is not None]
            
            if not valid_points:
                self.logger.warning("No valid points to upsert")
                return False

            self.client.upsert(collection_name=collection_name, points=valid_points)
            self.logger.info(f"Successfully upserted {len(valid_points)} points to {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error upserting points: {e}")
            return False

    def search_by_text(self, collection_name: str, query_embedding: np.ndarray, 
                      limit: int = 5) -> List[Dict[str, Any]]:
        """Search images by text embedding"""
        try:
            query_vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=("text", query_vec),
                limit=limit
            )
            
            return [
                {
                    "file_name": result.payload.get("name"),
                    "score": result.score,
                    "category": result.payload.get("category"),
                    "description": result.payload.get("description"),
                    "type": result.payload.get("type_of_category"),
                    "image_path": result.payload.get("image_path")
                }
                for result in results
            ]
        except Exception as e:
            self.logger.error(f"Error searching by text: {e}")
            return []