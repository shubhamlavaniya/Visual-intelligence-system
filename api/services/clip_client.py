import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Union
import logging
from api.core.config import settings

logger = logging.getLogger(__name__)

class CLIPClient:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        """Load the CLIP model and processor"""
        try:
            logger.info(f"Loading CLIP model: {settings.EMBEDDING_MODEL}")
            self.model = CLIPModel.from_pretrained(settings.EMBEDDING_MODEL)
            self.processor = CLIPProcessor.from_pretrained(settings.EMBEDDING_MODEL)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"CLIP model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image"""
        try:
            image = Image.open(image_path)
            
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
            return image_features.cpu().numpy().astype(np.float32).squeeze()
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding for {image_path}: {e}")
            raise

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
            return text_features.cpu().numpy().astype(np.float32).squeeze()
            
        except Exception as e:
            logger.error(f"Failed to generate text embedding for '{text}': {e}")
            raise

    def batch_process_images(self, image_paths: list) -> list:
        """Process multiple images in a batch (more efficient)"""
        try:
            
            images = [Image.open(path).convert("RGB") for path in image_paths]

            
            with torch.no_grad():
                print("--- CLIP CLIENT: RUNNING NEW VERSION WITH PADDING ---") 
                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
            return image_features.cpu().numpy().astype(np.float32)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

# Create a global instance
clip_client = CLIPClient()