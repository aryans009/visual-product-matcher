import os
from django.core.asgi import get_asgi_application
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'visual_product_matcher.settings')
application = get_asgi_application()