import streamlit as st
import requests
import time
from PIL import Image
import io
import base64

# Configuration
API_URL = "http://api:8000"  # For Docker compose
# API_URL = "http://localhost:8000"  # For local development

def main():
    st.set_page_config(
        page_title="AI Visual Search System",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
    }
    .image-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header"> AI-Powered Visual Search</h1>', unsafe_allow_html=True)
    st.markdown("""
    **Search through your image collection using natural language.**  
    The AI will find relevant images and explain why they match your query.
    """)
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g., 'a happy dog playing in the park', 'modern architecture with glass windows'",
            label_visibility="collapsed"
        )
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=10, value=5, help="Number of results to show")
    
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Display results
    if search_button and query:
        with st.spinner("🔍 Searching for relevant images..."):
            try:
                # Call the search API
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/search",
                    json={"query": query, "top_k": top_k},
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    processing_time = results["processing_time"]
                    
                    st.success(f"Found {len(results['results'])} results in {processing_time:.2f} seconds")
                    
                    # Display results in a grid
                    for i, result in enumerate(results["results"], 1):
                        with st.container():
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display image
                                try:
                                    image_url = f"{API_URL}{result['image_url']}"
                                    image_response = requests.get(image_url, timeout=10)
                                    
                                    if image_response.status_code == 200:
                                        image = Image.open(io.BytesIO(image_response.content))
                                        st.image(
                                            image,
                                            caption=f"Result {i}",
                                            use_column_width=True
                                        )
                                    else:
                                        st.error("Failed to load image")
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                            
                            with col2:
                                # Display result information
                                st.markdown(f"### Result {i}")
                                st.markdown(f"**Filename:** `{result['filename']}`")
                                st.markdown(f"**Relevance score:** <span class='score-badge'>{result['score']:.3f}</span>", 
                                           unsafe_allow_html=True)
                                
                                # Display AI explanation
                                if result['explanation']:
                                    st.markdown("** AI Explanation:**")
                                    st.info(result['explanation'])
                                else:
                                    st.warning("No explanation available")
                            
                            st.divider()
                
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the search API: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    
    # Sidebar with system information
    with st.sidebar:
        st.header("System Information")
        
        # Health check
        try:
            health_response = requests.get(f"{API_URL}/api/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("System is healthy")
                st.write(f"**Qdrant Connected:** {'✅' if health_data['qdrant_connected'] else '❌'}")
                st.write(f"**CLIP Model Loaded:** {'✅' if health_data['clip_model_loaded'] else '❌'}")
            else:
                st.error("API is not responding")
        except:
            st.error("Cannot connect to API")
        
        st.divider()
        st.header("How to Use")
        st.markdown("""
        1. Enter a natural language description of what you want to find
        2. Click the Search button
        3. View the results with AI explanations
        
        **Examples:**
        - "sunset over mountains"
        - "people laughing together"
        - "red sports car"
        - "cozy coffee shop interior"
        """)
        
        st.divider()
        st.header("About")
        st.markdown("""
        This visual search system uses:
        - **CLIP** for image and text embeddings
        - **Qdrant** for vector similarity search
        - **GPT-4o-mini** for AI explanations
        - **FastAPI** for the backend
        - **Streamlit** for the frontend
        """)

if __name__ == "__main__":
    main()