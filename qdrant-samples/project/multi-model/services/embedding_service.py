from PIL import Image, UnidentifiedImageError
from transformers import AutoFeatureExtractor, AutoModel
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional
from utils.logger import Logger

class EmbeddingService:
    """Handle image and text embeddings"""
    
    def __init__(self):
        self.logger = Logger()
        try:
            self.image_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.image_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding models: {e}")
            raise

    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate embedding for an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            features = self.image_extractor(images=img, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.image_model(**features)
            
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            return embedding
            
        except (UnidentifiedImageError, FileNotFoundError):
            self.logger.warning(f"Cannot open image file: {image_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error generating image embedding for {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        try:
            embedding = self.text_model.encode(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating text embedding: {e}")
            return None
