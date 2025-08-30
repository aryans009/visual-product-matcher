Visual Product Matcher
A sophisticated web application designed to find visually similar products using machine learning. Users can upload an image or provide a URL, and the application will return a ranked list of the most similar items from its database. The project is built with Django, leverages a Hugging Face Vision Transformer model for feature extraction, and uses Amazon S3 for scalable cloud-based image storage.

Key Features
Dual Image Input: Users can either upload an image file directly from their device or paste a URL to an image online.

AI-Powered Similarity Search: Utilizes the google/vit-base-patch16-224-in21k model to convert images into feature vectors, enabling accurate visual comparisons.

Dynamic Filtering: Results can be filtered in real-time by a similarity score, allowing users to narrow down matches.

Scalable Cloud Storage: All product and user-uploaded images are stored and served from an Amazon S3 bucket, ensuring performance and reliability.

Database Seeding: Comes with a management command to automatically populate the database with a sample set of over 50 products.

Responsive UI: The front-end is designed to be clean, intuitive, and fully responsive for use on both desktop and mobile devices.

Production-Ready: Configured for a production environment with secure key management, an efficient Gunicorn web server, and WhiteNoise for static file serving.

Technical Stack & Requirements
Backend: Django

Machine Learning: PyTorch, Hugging Face transformers

Image Processing: Pillow (PIL)

Cloud Storage: Amazon S3

Python Libraries: boto3, django-storages, python-dotenv, requests, gunicorn, whitenoise

Database: SQLite (for development), compatible with PostgreSQL (for production)

Frontend: HTML, CSS (no framework)

Directory Structure
The project is organized into a main configuration directory and a core matcher app, which contains the primary application logic.

visual_product_matcher/
├── .env                  # Stores all secret keys (MUST NOT be committed to Git)
├── manage.py             # Django's command-line utility
├── README.md             # This file
├── requirements.txt      # Lists all Python dependencies for pip
├── static/               # Static files (CSS, JS) for the project
│   └── css/
│       └── style.css
├── templates/            # Base HTML templates
│   └── base.html
├── db.sqlite3            # The SQLite database file (for local development)
│
├── matcher/              # The core application
│   ├── __init__.py
│   ├── admin.py          # Admin site configuration
│   ├── apps.py
│   ├── migrations/       # Database migration files
│   ├── models.py         # Defines the Product and UploadedImage database models
│   ├── similarity.py     # Core ML logic for feature extraction and comparison
│   ├── tests.py
│   ├── urls.py           # URL routing for the matcher app
│   ├── views.py          # Handles web requests and application logic
│   └── management/       # Custom Django management commands
│       └── commands/
│           └── populate_products.py # Script to seed the database
│
└── visual_product_matcher/ # Django project configuration
    ├── __init__.py
    ├── asgi.py
    ├── settings.py       # Main project settings (keys, apps, S3 config)
    ├── urls.py           # Root URL configuration
    └── wsgi.py           # WSGI entry-point for the production server

Local Setup and Installation
1. Prerequisites
Python 3.9+

An AWS Account with an active S3 bucket.

AWS IAM credentials (Access Key ID and Secret Access Key) with S3 access.

2. Clone the Repository
git clone <your-repository-url>
cd visual_product_matcher

3. Set Up Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install Dependencies
pip install -r requirements.txt

5. Configure Environment Variables
Create a .env file in the project root. This file is critical for security and must be included in your .gitignore.

# .env file

SECRET_KEY='your-unique-django-secret-key'
DEBUG=True

AWS_ACCESS_KEY_ID='YOUR_S3_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY='YOUR_S3_SECRET_ACCESS_KEY'
AWS_STORAGE_BUCKET_NAME='your-s3-bucket-name'
AWS_S3_REGION_NAME='your-s3-bucket-region'

6. Configure S3 Bucket Permissions
For images to be publicly visible on your website, you must configure two policies in your AWS S3 Console.

Bucket Policy: Go to your S3 bucket > "Permissions" > "Bucket policy" and add the following to allow public reads (replace your-s3-bucket-name):

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::your-s3-bucket-name/*"
        }
    ]
}

CORS Configuration: On the same page, scroll to "Cross-origin resource sharing (CORS)" and add this policy to allow your local server to request images:

[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET"],
        "AllowedOrigins": ["[http://127.0.0.1:8000](http://127.0.0.1:8000)", "http://localhost:8000"],
        "ExposeHeaders": []
    }
]

7. Run Database Migrations
Apply the database schema to create your tables:

python manage.py migrate

8. Populate the Database
This command will download sample product data and upload all images to your configured S3 bucket.

python manage.py populate_products

9. Run the Server
Start the local development server:

python manage.py runserver

Navigate to http://127.0.0.1:8000 in your browser to use the application.