import os
from dotenv import load_dotenv

# Loading environment variables from .env file
load_dotenv()

class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EXPLANATION_MODEL: str = os.getenv("EXPLANATION_MODEL_NAME", "gpt-4o-mini")
    
    # Qdrant Configuration
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME: str = "image_embeddings"
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL_NAME", "openai/clip-vit-large-patch14")
    EMBEDDING_SIZE: int = int(os.getenv("EMBEDDING_SIZE", 768))
    
    # Paths - Handle Docker vs local development
    IMAGE_DIR: str = os.getenv("IMAGE_DIR", "data/images")
    DOCKER_IMAGE_DIR: str = "/app/images"  # Path inside Docker container
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    def get_image_dir(self):
        """Get the correct image directory path based on environment"""
        # Check if we're running in Docker by checking if the Docker path exists
        if os.path.exists(self.DOCKER_IMAGE_DIR):
            return self.DOCKER_IMAGE_DIR
        return self.IMAGE_DIR

# Create a global settings instance
settings = Settings()