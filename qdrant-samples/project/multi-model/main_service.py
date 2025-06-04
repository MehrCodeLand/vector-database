from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService
from services.animal_service import AnimalService
from utils.logger import Logger
from typing import List, Dict, Any, Optional

class ImageSearchService:
    """Main service orchestrating all operations"""
    
    def __init__(self, collection_name: str = "animal_image_search", 
                 images_directory: str = "images/cats"):
        self.collection_name = collection_name
        self.images_directory = images_directory
        
        # Initialize services
        self.logger = Logger()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.animal_service = AnimalService(images_directory)

    def initialize_database(self) -> bool:
        """Initialize the vector database with animal data"""
        try:
            # Create collection
            if not self.qdrant_service.create_collection(self.collection_name):
                return False

            # Get animals from directory
            animals = self.animal_service.create_animals_from_directory()
            if not animals:
                self.logger.error("No animals found in directory")
                return False

            # Create embeddings and points
            points = []
            for i, animal in enumerate(animals):
                # Generate embeddings
                image_embedding = self.embedding_service.get_image_embedding(animal.image_path)
                text_for_embedding = f"{animal.category} {animal.type_of_category} {animal.description}"
                text_embedding = self.embedding_service.get_text_embedding(text_for_embedding)

                # Create point
                point = self.qdrant_service.create_point(i, image_embedding, text_embedding, animal)
                if point:
                    points.append(point)

            # Upsert points to database
            success = self.qdrant_service.upsert_points(self.collection_name, points)
            if success:
                self.logger.info(f"Successfully initialized database with {len(points)} animals")
            return success

        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            return False

    def search_by_query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search animals by text query"""
        try:
            query_embedding = self.embedding_service.get_text_embedding(query)
            if query_embedding is None:
                return []

            results = self.qdrant_service.search_by_text(
                self.collection_name, query_embedding, limit
            )
            return results
        except Exception as e:
            self.logger.error(f"Error searching by query '{query}': {e}")
            return []