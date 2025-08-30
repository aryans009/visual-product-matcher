import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

# We are now using a much smaller "tiny" version of the Vision Transformer because render was not able to handle and demanding to pay for larger models.
MODEL_NAME = 'WinKawaks/vit-tiny-patch16-224'

try:
    image_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTModel.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    image_processor = None
    model = None

def extract_features(image: Image.Image) -> np.ndarray | None:
    """
    Extracts a feature vector from an image using the ViT model.
    """
    if not model or not image_processor:
        print("Model or image processor not loaded.")
        return None
        
    try:
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        features = last_hidden_states.mean(dim=1).cpu().numpy()
        return features
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        return None

def find_similar_products(uploaded_features, all_products, top_n=20):
    """
    Finds similar products by comparing feature vectors using cosine similarity.
    """
    if uploaded_features is None or not all_products:
        return []

    product_features = np.array([np.frombuffer(p.feature_vector, dtype=np.float32) for p in all_products])
    
    if product_features.ndim == 1:
        product_features = product_features.reshape(1, -1)
    if uploaded_features.ndim == 1:
        uploaded_features = uploaded_features.reshape(1, -1)

    similarities = cosine_similarity(uploaded_features, product_features)[0]

    similar_products_with_scores = []
    for i, product in enumerate(all_products):
        similar_products_with_scores.append({
            'product': product,
            'similarity': similarities[i]
        })

    similar_products_with_scores.sort(key=lambda x: x['similarity'], reverse=True)

    return similar_products_with_scores[:top_n]
