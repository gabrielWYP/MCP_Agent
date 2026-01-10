from typing import TypedDict

class MLOpsState(TypedDict):
    """Estado compartido entre todos los nodos del grafo."""
    new_data_path: str                          # Ubicación de nuevos datos
    model_version: int                          # Versión actual del modelo
    candidate_model_path: str                   # Ruta del modelo candidato
    candidate_metrics: dict                     # Métricas del modelo candidato
    production_metrics: dict                    # Métricas del modelo en producción
    deployment_decision: str                    # "APPROVE", "REJECT", "HUMAN_REVIEW"
    failure_analysis_report: str                # Reporte de fallos
    data_quality_score: float                   # Score de calidad (0-10)
    analysis_reason: str                        # Razonamiento de la decisión
    timestamp: str                              # Timestamp de operaciones
