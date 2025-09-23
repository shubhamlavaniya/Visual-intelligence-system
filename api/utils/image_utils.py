import os
from PIL import Image
import numpy as np
from api.core.config import settings

def get_image_files() -> list: 
    """Get all image files from the configured image directory"""
     # Using the configured directory
    image_dir = settings.get_image_dir() 
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, filename))
    
    return sorted(image_files)

def validate_image(image_path: str) -> bool:
    """Validate if a file is a valid image"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def get_image_id_from_path(image_path: str) -> str:
    """Extract image ID from file path"""
    filename = os.path.basename(image_path)
    return os.path.splitext(filename)[0]

def get_image_path(filename: str) -> str:
    """Get full path to an image file"""
    return os.path.join(settings.get_image_dir(), filename)