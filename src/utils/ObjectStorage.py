import os
import sys
import s3fs
from typing import Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
import json
import pandas as pd

# --- Definición del Estado del Grafo ---
class MLOpsState(TypedDict):
    new_data_path: str
    model_version: int
    candidate_model_path: str
    candidate_metrics: Dict[str, Any]
    production_metrics: Dict[str, Any]
    deployment_decision: str
    failure_analysis_report: str

# --- Configuración de S3FS ---
# Estas variables DEBEN cargarse desde el entorno.
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME") # Puedes sobrescribir el nombre del bucket si es necesario


def get_s3fs():
    """Inicializa y devuelve un sistema de archivos s3fs."""
    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY]):
        raise ValueError("Las variables de entorno S3_ENDPOINT_URL, S3_ACCESS_KEY y S3_SECRET_KEY deben estar configuradas.")
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_ENDPOINT_URL},
        key=S3_ACCESS_KEY,
        secret=S3_SECRET_KEY,
    )

def count_and_list_objects(bucket_name: str) -> list[str]:
    """
    Lista todos los objetos en el bucket especificado.
    """
    print(f"Intentando listar objetos en el bucket: {bucket_name}")
    s3 = get_s3fs()
    try:
        # s3.ls() devuelve una lista de rutas completas (ej. 'bucket/folder/file.txt')
        # Para listar todo el contenido del bucket, pasamos el nombre del bucket como ruta base.
        # El argumento 'detail=False' devuelve solo los nombres de los archivos/directorios.
        objects = s3.ls(bucket_name, detail=False)
        # Filtramos para mostrar solo los objetos, no el propio bucket si aparece
        
        lista_objetos = [obj for obj in objects if obj != bucket_name]
        
        print(f"Objetos encontrados en el bucket '{bucket_name}': {lista_objetos}")
        
        return len(lista_objetos), lista_objetos
    
    except Exception as e:
        print(f"ERROR al listar objetos en el bucket '{bucket_name}': {e}")
        raise