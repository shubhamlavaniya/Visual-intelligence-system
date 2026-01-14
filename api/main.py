from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import time
import os
import logging
from typing import List

from api.core.config import settings
from api.core.models import SearchRequest, SearchResponse, SearchResult, HealthResponse
from api.services.qdrant_service import qdrant_service
from api.services.clip_client import clip_client
from api.services.explanation_generator import explanation_generator

from api.utils.image_utils import get_image_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Visual Search System API",
    description="Enterprise-grade visual search with AI explanations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images
app.mount("/images", StaticFiles(directory=settings.get_image_dir()), name="images")

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        qdrant_info = qdrant_service.get_collection_info()
        qdrant_connected = qdrant_info is not None
    except Exception:
        qdrant_connected = False
    
    return HealthResponse(
        status="healthy",
        qdrant_connected=qdrant_connected,
        clip_model_loaded=clip_client.model is not None
    )

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_images(request: SearchRequest):
    """
    Search for images based on natural language query.
    Returns top matching images with AI explanations.
    """
    start_time = time.time()
    
    try:
        # Generate embedding for the text query
        logger.info(f"Processing query: {request.query}")
        query_embedding = clip_client.get_text_embedding(request.query)
        
        # Search for similar images in Qdrant
        search_results = qdrant_service.search_similar(query_embedding, top_k=request.top_k)
        
        # Prepare results with explanations
        results = []
        for result in search_results:
            image_id = str(result["id"])
            filename = result["payload"]["filename"]
            #image_path = os.path.join(settings.IMAGE_DIR, filename)
            
            image_path = get_image_path(filename)
            
            # Generate AI explanation
            explanation = explanation_generator.generate_explanation(image_path, request.query)
            
            # Create result object
            search_result = SearchResult(
                image_id=image_id,
                filename=filename,
                score=result["score"],
                explanation=explanation,
                image_url=f"/images/{filename}"
            )
            results.append(search_result)
        
        processing_time = time.time() - start_time
        logger.info(f"Search completed in {processing_time:.2f} seconds")
        
        return SearchResponse(
            results=results,
            query=request.query,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/image/{filename}", tags=["Images"])
async def get_image(filename: str):
    """Serve an image file"""
    image_path = os.path.join(settings.IMAGE_DIR, filename)

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def api_health():
    """API health check"""
    return await health_check()

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up Visual Search System API...")
    # Services are already initialized as global instances

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Visual Search System API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)