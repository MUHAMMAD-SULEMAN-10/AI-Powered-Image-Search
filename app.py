import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import os
import glob
from pathlib import Path
import zipfile
import urllib.request

# Page configuration
st.set_page_config(
    page_title="AI Image Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .result-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the CLIP model"""
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_data


def download_and_extract_images():
    """Download and extract the Unsplash dataset"""
    zip_path = "unsplash-25k-photos.zip"
    photos_dir = "photos"
    
    if not os.path.exists(photos_dir):
        if not os.path.exists(zip_path):
            with st.spinner("Downloading image dataset (this may take a few minutes)..."):
                url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/unsplash-25k-photos.zip"
                urllib.request.urlretrieve(url, zip_path)
        
        with st.spinner("Extracting images..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(photos_dir)

    
    return True

@st.cache_data
def load_image_paths(limit=2000):
    """Load image paths from the photos directory"""
    img_names = list(glob.glob('photos/*.jpg'))[:limit]
    return img_names

@st.cache_data
def encode_images(_model, img_names):
    """Encode all images into embeddings"""
    images = []
    valid_paths = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, img_path in enumerate(img_names):
        try:
            images.append(Image.open(img_path))
            valid_paths.append(img_path)
        except Exception as e:
            st.warning(f"Could not load {img_path}: {e}")
        
        progress_bar.progress((idx + 1) / len(img_names))
        status_text.text(f"Loading images: {idx + 1}/{len(img_names)}")
    
    status_text.text("Encoding images with CLIP model...")
    img_embed = _model.encode(
        images, 
        batch_size=32, 
        convert_to_tensor=True, 
        show_progress_bar=False
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return img_embed, valid_paths

def search_images(model, query, img_embed, img_names, k=9, query_type="text"):
    """Search for images based on text query or image"""
    if query_type == "text":
        query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    else:  # image query
        query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    hits = util.semantic_search(query_emb, img_embed, top_k=k)[0]
    return hits

def main():
    st.title("🔍 AI-Powered Image Search")
    st.markdown("Search through 2000 images using natural language or upload an image to find similar ones!")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        num_results = st.slider("Number of results", min_value=3, max_value=20, value=9, step=1)
        st.markdown("---")
        st.markdown("### About")
        st.info("This app uses CLIP (Contrastive Language-Image Pre-training) to search images using text descriptions or similar images.")
    
    # Initialize
    try:
        model = load_model()
        
        # Check if dataset is ready
        if 'dataset_ready' not in st.session_state:
            download_and_extract_images()
            st.session_state.dataset_ready = True
        
        # Load and encode images
        if 'img_embed' not in st.session_state:
            with st.spinner("Loading and encoding images... This will take a moment on first run."):
                img_names = load_image_paths(limit=2000)
                if not img_names:
                    st.error("No images found! Please check the photos directory.")
                    return
                
                img_embed, valid_paths = encode_images(model, img_names)
                st.session_state.img_embed = img_embed
                st.session_state.img_names = valid_paths
                st.success(f"✅ Loaded {len(valid_paths)} images successfully!")
        
        img_embed = st.session_state.img_embed
        img_names = st.session_state.img_names
        
        # Search interface
        st.markdown("---")
        search_type = st.radio("Search by:", ["Text Description", "Similar Image"], horizontal=True)
        
        if search_type == "Text Description":
            query = st.text_input("Enter your search query:", placeholder="e.g., 'dog playing in park', 'sunset over mountains'")
            
            if st.button("🔍 Search") and query:
                with st.spinner("Searching..."):
                    hits = search_images(model, query, img_embed, img_names, k=num_results, query_type="text")
                
                st.markdown(f"### Results for: *{query}*")
                
                # Display results in grid
                cols = st.columns(3)
                for idx, hit in enumerate(hits):
                    col = cols[idx % 3]
                    img_path = img_names[hit['corpus_id']]
                    score = hit['score']
                    
                    with col:
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                            st.caption(f"Score: {score:.4f}")
                            st.caption(f"📁 {os.path.basename(img_path)}")
                        except Exception as e:
                            st.error(f"Could not display image: {e}")
        
        else:  # Similar Image search
            uploaded_file = st.file_uploader("Upload an image to find similar ones:", type=['jpg', 'jpeg', 'png'])
            
            if uploaded_file is not None:
                query_img = Image.open(uploaded_file)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### Your Image:")
                    st.image(query_img, use_container_width=True)
                
                with col2:
                    if st.button("🔍 Find Similar Images"):
                        with st.spinner("Searching for similar images..."):
                            hits = search_images(model, query_img, img_embed, img_names, k=num_results, query_type="image")
                        
                        st.markdown("### Similar Images Found:")
                
                if st.session_state.get('show_similar_results'):
                    cols = st.columns(3)
                    for idx, hit in enumerate(hits):
                        col = cols[idx % 3]
                        img_path = img_names[hit['corpus_id']]
                        score = hit['score']
                        
                        with col:
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                                st.caption(f"Similarity: {score:.4f}")
                                st.caption(f"📁 {os.path.basename(img_path)}")
                            except Exception as e:
                                st.error(f"Could not display image: {e}")
                
                if 'hits' in locals():
                    st.session_state.show_similar_results = True
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()