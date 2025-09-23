import base64
import logging
from openai import OpenAI
from PIL import Image
import io
from typing import Optional
from api.core.config import settings



logger = logging.getLogger(__name__)

class ExplanationGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.EXPLANATION_MODEL

    def generate_explanation(self, image_path: str, query: str) -> Optional[str]:
        """
        Generate an explanation for why the image is relevant to the query using GPT-4o-mini.
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            # Create the prompt
            prompt = self._create_prompt(query)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=150,
                temperature=0.2  # Lower temperature for more factual responses
            )
            
            explanation = response.choices[0].message.content
            logger.info(f"Generated explanation for {image_path}: {explanation}")
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for {image_path}: {e}")
            return None

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize for efficiency (optional, but reduces token usage)
                img.thumbnail((512, 512))
                
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _create_prompt(self, query: str) -> str:
        """Create the prompt for the LLM"""
        return f"""
        You are an AI visual search analyst. Explain why this image is a relevant result for the user's query: '{query}'.

        Your explanation must be:
        - Concise (1-2 sentences)
        - Business-appropriate and technically sound
        - Refer to specific visual elements (objects, colors, actions)
        - Mention attributes (style, lighting, composition)
        - Describe the context (setting, scene, atmosphere)

        Focus on the most relevant aspects that connect the image to the query.
        Do not use markdown. Just return the explanation.
        """

# Create a global instance
explanation_generator = ExplanationGenerator()