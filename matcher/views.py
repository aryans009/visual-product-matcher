import requests
from io import BytesIO
from django.shortcuts import render
from django.core.files.base import ContentFile
from .models import Product, UploadedImage
from .similarity import extract_features, find_similar_products
from PIL import Image as PILImage
import numpy as np

def index(request):
    similar_products = []
    uploaded_image_url = None
    error_message = None
    loading = False

    if request.method == 'POST':
        loading = True
        image_file = request.FILES.get('image_file')
        image_url_from_post = request.POST.get('image_url')
        
        image_to_process = None
        image_bytes = None
        image_filename = 'uploaded_image.jpg' 

        try:
            if image_file:
                image_filename = image_file.name
                image_bytes = image_file.read()
                image_to_process = PILImage.open(BytesIO(image_bytes)).convert("RGB")

            elif image_url_from_post:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(image_url_from_post, headers=headers, timeout=15)
                response.raise_for_status()
                image_bytes = response.content
                image_to_process = PILImage.open(BytesIO(image_bytes)).convert("RGB")
                image_filename = image_url_from_post.split('/')[-1]

            else:
                error_message = "Please upload an image or provide a URL."

            if image_to_process and image_bytes:
                uploaded_image_instance = UploadedImage()
                uploaded_image_instance.image.save(image_filename, ContentFile(image_bytes))
                
                uploaded_image_url = uploaded_image_instance.image.url
                
                uploaded_features = extract_features(image_to_process)

                if uploaded_features is not None:
                    all_products = list(Product.objects.exclude(feature_vector__isnull=True))
                    similar_products = find_similar_products(uploaded_features, all_products)

                    min_similarity = request.POST.get('similarity_score')
                    if min_similarity:
                        min_similarity = float(min_similarity)
                        similar_products = [p for p in similar_products if p['similarity'] >= min_similarity]
                else:
                    error_message = "Could not extract features from the image."

        except PILImage.UnidentifiedImageError:
            error_message = "Could not identify the file as an image. Please check the file or URL."
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to retrieve image from URL: {e}"
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
        
        loading = False

    return render(request, 'matcher/index.html', {
        'similar_products': similar_products,
        'uploaded_image_url': uploaded_image_url,
        'error_message': error_message,
        'loading': loading,
    })

