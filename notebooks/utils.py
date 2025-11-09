
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

def list_all_bucket_objects(bucket_name: str) -> list[str]:
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
        return [obj for obj in objects if obj != bucket_name]
    except Exception as e:
        print(f"ERROR al listar objetos en el bucket '{bucket_name}': {e}")
        raise

# --- Nodos del Grafo ---

def check_for_new_data(state: MLOpsState) -> MLOpsState:
    """
    Nodo 1: Verifica si hay nuevos datos en el bucket de Object Storage.
    """
    print("--- (1) NODO: CHECK_FOR_NEW_DATA ---")
    try:
        s3 = get_s3fs()
        # Definimos una carpeta "new" para buscar imágenes nuevas
        target_path = f"{BUCKET_NAME}/new/"
        
        if s3.exists(target_path) and len(s3.ls(target_path)) > 0:
            print(f"--> Se encontraron archivos nuevos en 's3://{target_path}'.")
            state["new_data_path"] = f"s3://{target_path}"
            state["failure_analysis_report"] = None
        else:
            print(f"--> No se encontraron archivos nuevos en 's3://{target_path}'. Finalizando ciclo.")
            # Usamos una ruta especial para indicar que no hay nada que hacer
            state["new_data_path"] = "NO_NEW_DATA"
            
    except Exception as e:
        print(f"ERROR al conectar con Object Storage: {e}")
        state["new_data_path"] = "NO_NEW_DATA"
        state["failure_analysis_report"] = f"Error en check_for_new_data: {e}"

    return state

def validate_data(state: MLOpsState) -> MLOpsState:
    """
    Nodo 2: Analiza la calidad de los datos.
    """
    print("--- (2) NODO: VALIDATE_DATA ---")
    # Lógica de simulación
    print("Aquí llamaremos a un VLM para analizar la calidad de las imágenes.")
    state["failure_analysis_report"] = None # Asumimos que es buena
    return state

def run_training_job(state: MLOpsState) -> MLOpsState:
    """
    Nodo 3: Ejecuta el trabajo de entrenamiento.
    """
    print("--- (3) NODO: RUN_TRAINING_JOB ---")
    # Lógica de simulación
    print("Este nodo ejecutará el script de entrenamiento del modelo de DL.")
    state["candidate_model_path"] = "/path/to/candidate_model.tflite"
    state["candidate_metrics"] = {"accuracy": 0.92, "f1_maduro": 0.95}
    state["production_metrics"] = {"accuracy": 0.91, "f1_maduro": 0.88}
    return state

def analyze_performance(state: MLOpsState) -> MLOpsState:
    """
    Nodo 4: Analiza el rendimiento del modelo candidato.
    """
    print("--- (4) NODO: ANALYZE_PERFORMANCE ---")
    # Lógica de simulación
    print("Un LLM analizará si el nuevo modelo es mejor que el de producción.")
    state["deployment_decision"] = "APPROVE"
    return state

def deploy_to_production(state: MLOpsState) -> MLOpsState:
    """
    Nodo 7: Despliega el modelo a producción.
    """
    print("--- (7) NODO: DEPLOY_TO_PRODUCTION ---")
    print("Aquí iría la lógica para desplegar el modelo.")
    return state

def alert_human(state: MLOpsState) -> MLOpsState:
    """
    Nodo 6: Envía una alerta a un operador humano.
    """
    print("--- (6) NODO: ALERT_HUMAN ---")
    report = state.get("failure_analysis_report")
    if report:
        print(f"Alerta de Calidad de Datos: {report}")
    else:
        print("Alerta de Despliegue: El modelo fue rechazado.")
    return state