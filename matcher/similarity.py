import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from io import BytesIO

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_image_from_url(url):
    """Downloads an image from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

def extract_features(image):
    """Extracts a feature vector from an image."""
    if image is None:
        return None
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def find_similar_products(uploaded_image_features, all_products, top_n=10):
    """Finds similar products based on feature vectors."""
    if uploaded_image_features is None or not all_products:
        return []

    product_features = np.array([np.frombuffer(p.feature_vector, dtype=np.float32) for p in all_products])
    
    similarities = cosine_similarity(uploaded_image_features, product_features)[0]
    
    # Get indices of top N similar products
    similar_indices = np.argsort(similarities)[-top_n:][::-1]
    
    similar_products = []
    for i in similar_indices:
        product = all_products[i]
        similarity_score = similarities[i]
        similar_products.append({'product': product, 'similarity': similarity_score})
        
    return similar_products
