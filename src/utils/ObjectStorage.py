import os
import sys
import s3fs
from pathlib import Path
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
BUCKET_NAME = os.getenv("BUCKET_NAME")


def get_s3fs():
    """Inicializa y devuelve un sistema de archivos s3fs."""
    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY]):
        raise ValueError("Las variables de entorno S3_ENDPOINT_URL, S3_ACCESS_KEY y S3_SECRET_KEY deben estar configuradas.")
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": S3_ENDPOINT_URL},
        key=S3_ACCESS_KEY,
        secret=S3_SECRET_KEY,
    )

def download_object(key: str, local_path: str) -> bool:
    """
    Download a single object from OCI/S3 to local disk.

    Uses size-based dedup: if local file exists with matching size, skip download.

    Args:
        key: Full object key in the bucket (e.g. 'bucket/news/rgb/img001.jpg')
        local_path: Local destination path

    Returns:
        True if downloaded, False if skipped (cache hit)
    """
    s3 = get_s3fs()
    local = Path(local_path)

    # Check cache hit by size
    if local.exists():
        try:
            remote_info = s3.info(key)
            remote_size = remote_info.get("size", remote_info.get("Size", -1))
            if remote_size == local.stat().st_size:
                return False  # cache hit
        except Exception:
            pass  # info failed, proceed with download

    # Ensure parent directory exists
    local.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp file, then rename (atomic, prevents partial files)
    tmp_path = local.with_suffix(local.suffix + ".tmp")
    try:
        s3.get(key, str(tmp_path))
        tmp_path.rename(local)
    except Exception:
        # Clean up partial download
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return True


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