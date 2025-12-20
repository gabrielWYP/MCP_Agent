import os
import json
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno desde .env en el root del proyecto
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(ENV_PATH), override=True)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")