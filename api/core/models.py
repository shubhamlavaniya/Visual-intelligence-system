from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    image_id: str
    filename: str
    score: float
    explanation: Optional[str] = None
    image_url: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    clip_model_loaded: bool

class ExplanationRequest(BaseModel):
    image_id: str
    query: str