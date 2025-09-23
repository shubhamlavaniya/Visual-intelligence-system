import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient, models
import logging

# --- Configuration ---
# All settings are now in one place for clarity
class Config:
    IMAGE_DIR = "data/images"
    EMBEDDING_MODEL = "openai/clip-vit-large-patch14"
    EMBEDDING_SIZE = 768
    COLLECTION_NAME = "image_embeddings"
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333

def main():
    """Main function to generate and upsert embeddings."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting embedding generation script...")

    try:
        # --- 1. Initialize Qdrant Client ---
        logger.info(f"Connecting to Qdrant at {Config.QDRANT_HOST}:{Config.QDRANT_PORT}...")
        client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT, timeout=60)

        # --- 2. Set up Qdrant Collection ---
        logger.info(f"Setting up collection '{Config.COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=models.VectorParams(size=Config.EMBEDDING_SIZE, distance=models.Distance.COSINE)
        )
        logger.info(" Collection is ready.")

        # --- 3. Load CLIP Model ---
        logger.info(f"Loading CLIP model '{Config.EMBEDDING_MODEL}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(Config.EMBEDDING_MODEL)
        processor = CLIPProcessor.from_pretrained(Config.EMBEDDING_MODEL)
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on '{device}'.")

        # --- 4. Process Images and Upsert to Qdrant ---
        image_files = [f for f in os.listdir(Config.IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not image_files:
            logger.error(" No images found in the directory. Exiting.")
            return

        logger.info(f" Found {len(image_files)} images. Starting processing...")
        batch_size = 16
        point_id_counter = 0 # <-- Using an integer counter

        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
            batch_files = image_files[i:i + batch_size]
            batch_paths = [os.path.join(Config.IMAGE_DIR, f) for f in batch_files]
            batch_images, valid_files = [], []
            
            for file_path in batch_paths:
                try:
                    img = Image.open(file_path).convert("RGB")
                    batch_images.append(img)
                    valid_files.append(os.path.basename(file_path))
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid image {os.path.basename(file_path)}: {e}")
                    continue
            
            if not batch_images: continue

            with torch.no_grad():
                inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                image_features = model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy()

            points_to_upsert = [
                models.PointStruct(
                    id=point_id_counter + j, # <-- Assigning a unique integer ID
                    vector=embedding.tolist(),
                    payload={"filename": filename}
                ) for j, (filename, embedding) in enumerate(zip(valid_files, embeddings))
            ]

            client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=points_to_upsert,
                wait=True
            )
            point_id_counter += len(points_to_upsert)

        logger.info(f"\n Successfully processed and stored {point_id_counter} images.")

        # Use the count() method for an accurate, real-time vector count
        count_result = client.count(
            collection_name=Config.COLLECTION_NAME,
            exact=True  # Performs an exact count
)
        logger.info(f" Qdrant collection now contains {count_result.count} vectors.")

    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()