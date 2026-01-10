"""
MLOps Graph Nodes para orquestación del pipeline de detección de madurez de mangos.
Todos los nodos implementados en un solo archivo.
"""

from datetime import datetime
from typing import Literal
import json
from utils.logger import logger_singleton as logger
from storage_logic.logic import check_new_objects
from variables.configvariables import BUCKET_NAME
from state import MLOpsState


# =====================================================================
# NODO 1: VERIFICAR NUEVOS DATOS
# =====================================================================

def check_for_new_data(state: MLOpsState) -> MLOpsState:
    """
    Node 1: Verifica si hay nuevos datos en el almacenamiento en la nube.
    
    Acción: Monitorea el bucket S3/OCI para detectar nuevas imágenes de mangos.
    
    Lógica de enrutamiento:
    - Si hay nuevos datos → procede a validate_data
    - Si no hay nuevos datos → termina y espera
    """
    logger.info("🔍 Node 1: Verificando nuevos datos...")
    
    try:
        have_news = check_new_objects(BUCKET_NAME)
        
        if have_news:
            logger.info("✅ ¡Nuevos datos detectados!")
            state["new_data_path"] = f"s3://{BUCKET_NAME}/news"
            state["timestamp"] = datetime.now().isoformat()
            return state
        else:
            logger.info("⏸️ Sin nuevos datos. Orquestador en espera.")
            state["failure_analysis_report"] = "No se detectaron nuevos datos en almacenamiento."
            return state
            
    except Exception as e:
        logger.error(f"❌ Error al verificar nuevos datos: {str(e)}")
        state["failure_analysis_report"] = f"Error verificando datos: {str(e)}"
        return state


# =====================================================================
# NODO 2: VALIDAR CALIDAD DE DATOS
# =====================================================================

def validate_data(state: MLOpsState) -> MLOpsState:
    """
    Node 2: Valida la calidad de datos usando un Modelo de Lenguaje Visual (VLM).
    
    Acción: Llama a VLM (ej: Gemini 1.5 Pro) para:
    - Analizar muestras de imágenes de new_data_path
    - Verificar claridad, iluminación, corrupción
    - Retornar un quality_score (0-10)
    
    Lógica de enrutamiento:
    - Si quality_score >= 7.0 → procede a run_training_job
    - Si quality_score < 7.0 → alerta a operador humano
    """
    logger.info("🎨 Node 2: Validando calidad de datos con VLM...")
    
    try:
        # TODO: Integrar llamada real a Gemini API
        # response = gemini_client.analyze_images(state["new_data_path"])
        # quality_score = response["quality_score"]
        
        # Respuesta simulada para prototipado
        quality_score = 8.5  # Score alto de calidad
        
        logger.info(f"📊 Score de calidad: {quality_score}/10")
        state["data_quality_score"] = quality_score
        state["timestamp"] = datetime.now().isoformat()
        
        if quality_score >= 7.0:
            logger.info("✅ Calidad de datos aceptable. Procediendo al entrenamiento.")
            return state
        else:
            logger.warning("⚠️ Calidad insuficiente. Alertando operador.")
            state["failure_analysis_report"] = (
                f"Validación de datos fallida. Score: {quality_score}/10. "
                "Imágenes pueden estar borrosas, mal iluminadas o corruptas."
            )
            return state
            
    except Exception as e:
        logger.error(f"❌ Error validando datos: {str(e)}")
        state["failure_analysis_report"] = f"Error en validación de datos: {str(e)}"
        return state


# =====================================================================
# NODO 3: EJECUTAR TRABAJO DE ENTRENAMIENTO
# =====================================================================

def run_training_job(state: MLOpsState) -> MLOpsState:
    """
    Node 3: Ejecuta el trabajo de entrenamiento del modelo Deep Learning.
    
    Acción: Entrena el modelo (ej: MobileNetV2) con los datos validados
    - Entrena en los nuevos datos de mangos
    - Guarda artefacto del modelo (mango_model_v{N}.tflite)
    - Genera métricas de rendimiento
    
    Output:
    - candidate_model_path: Ruta del modelo entrenado
    - candidate_metrics: Métricas del nuevo modelo
    
    Transición: Siempre procede a analyze_performance
    """
    logger.info("🚀 Node 3: Ejecutando trabajo de entrenamiento...")
    
    try:
        # TODO: Llamar pipeline real de entrenamiento
        # from ml_pipeline.training.process import train_model
        # result = train_model(state["new_data_path"])
        
        new_version = state.get("model_version", 2) + 1
        state["model_version"] = new_version
        state["candidate_model_path"] = f"s3://{BUCKET_NAME}/models/mango_model_v{new_version}.tflite"
        state["candidate_metrics"] = {
            "accuracy": 0.90,
            "f1_class_maduro": 0.95,
            "f1_class_verde": 0.88,
            "f1_class_pinton": 0.92,
            "loss": 0.24
        }
        state["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"✅ Modelo v{new_version} entrenado exitosamente!")
        logger.info(f"📊 Métricas: {json.dumps(state['candidate_metrics'], indent=2)}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error durante entrenamiento: {str(e)}")
        state["failure_analysis_report"] = f"Fallo en entrenamiento: {str(e)}"
        return state


# =====================================================================
# NODO 4: ANALIZAR RENDIMIENTO CON LLM
# =====================================================================

def analyze_performance(state: MLOpsState) -> MLOpsState:
    """
    Node 4: Analiza y compara rendimiento del modelo usando un LLM.
    
    Acción: Un LLM (Claude, GPT, etc) analiza:
    - Métricas del modelo candidato (nuevo modelo v3)
    - Métricas del modelo en producción (modelo actual v2)
    - Objetivos de negocio (mejor detección de mangos maduros)
    
    Output:
    - deployment_decision: "APPROVE", "REJECT", o "HUMAN_REVIEW"
    - analysis_reason: Razonamiento detallado de la decisión
    
    Transición: Siempre procede a check_deployment_decision
    """
    logger.info("🧠 Node 4: Analizando rendimiento con LLM...")
    
    try:
        # Métricas de producción (valores por defecto)
        production_metrics = state.get("production_metrics", {
            "accuracy": 0.91,
            "f1_class_maduro": 0.88,
            "f1_class_verde": 0.90,
            "f1_class_pinton": 0.87,
            "loss": 0.26
        })
        
        candidate_metrics = state.get("candidate_metrics", {})
        
        # TODO: Integrar llamada real a LLM
        # prompt = f"""
        # Producción v{state['model_version']-1}: {production_metrics}
        # Candidato v{state['model_version']}: {candidate_metrics}
        # ¿Es el candidato mejor para identificar mangos maduros?
        # Retorna JSON con 'decision' ('APPROVE'/'REJECT') y 'reason'.
        # """
        # response = llm.invoke(prompt)
        
        # Lógica simulada: comparar F1 score de clase "maduro"
        candidate_f1_maduro = candidate_metrics.get("f1_class_maduro", 0)
        production_f1_maduro = production_metrics.get("f1_class_maduro", 0)
        
        if candidate_f1_maduro > production_f1_maduro:
            decision = "APPROVE"
            reason = (
                f"Aunque la precisión general bajó ligeramente, "
                f"F1 score para 'maduro' mejoró de {production_f1_maduro} "
                f"a {candidate_f1_maduro}. Crítico para objetivos de negocio."
            )
        else:
            decision = "REJECT"
            reason = (
                f"F1 score para 'maduro' bajó de {production_f1_maduro} "
                f"a {candidate_f1_maduro}. Inaceptable para producción."
            )
        
        state["deployment_decision"] = decision
        state["analysis_reason"] = reason
        state["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"📈 Decisión del análisis: {decision}")
        logger.info(f"📝 Razón: {reason}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error analizando rendimiento: {str(e)}")
        state["deployment_decision"] = "HUMAN_REVIEW"
        state["analysis_reason"] = f"Error durante análisis: {str(e)}"
        return state


# =====================================================================
# NODO 5: VERIFICAR DECISIÓN DE DESPLIEGUE (CONDICIONAL)
# =====================================================================

def check_deployment_decision(state: MLOpsState) -> Literal["deploy_to_production", "alert_human"]:
    """
    Node 5: Nodo condicional que enruta según la decisión de despliegue.
    
    Acción: Lee deployment_decision del estado y enruta:
    
    Lógica de enrutamiento:
    - Si decision == "APPROVE" → deploy_to_production
    - Si decision == "REJECT" o "HUMAN_REVIEW" → alert_human
    """
    logger.info("🔀 Node 5: Verificando decisión de despliegue...")
    
    decision = state.get("deployment_decision", "HUMAN_REVIEW")
    
    if decision == "APPROVE":
        logger.info("✅ Decisión: APPROVE - enrutando a despliegue")
        return "deploy_to_production"
    else:
        logger.warning(f"⚠️ Decisión: {decision} - enrutando a alerta humana")
        return "alert_human"


# =====================================================================
# NODO 6: ALERTAR AL OPERADOR HUMANO
# =====================================================================

def alert_human(state: MLOpsState) -> MLOpsState:
    """
    Node 6: Alerta al operador humano sobre rechazo o fallo.
    
    Acción: Envía notificación con:
    - Razón del rechazo o fallo
    - Comparación de métricas
    - Pasos recomendados
    
    Métodos de notificación (TODO):
    - Slack
    - Teams
    - Email
    
    Transición: Siempre termina el grafo
    """
    logger.info("📢 Node 6: Alertando operador humano...")
    
    decision = state.get("deployment_decision", "HUMAN_REVIEW")
    reason = state.get("analysis_reason", "")
    failure_report = state.get("failure_analysis_report", "")
    
    alert_message = f"""
    🛑 ALERTA MLOps - Intervención Manual Requerida
    
    Estado: {decision}
    Timestamp: {state.get('timestamp', 'N/A')}
    Versión del Modelo: v{state.get('model_version', 'N/A')}
    
    Razón:
    {reason or failure_report}
    
    Métricas del Candidato:
    {json.dumps(state.get('candidate_metrics', {}), indent=2)}
    
    Métricas de Producción:
    {json.dumps(state.get('production_metrics', {}), indent=2)}
    
    Por favor revisa y toma acción apropiada.
    """
    
    logger.warning(alert_message)
    
    # TODO: Implementar notificaciones reales
    # send_slack_notification(alert_message)
    # send_email_notification(alert_message)
    
    state["timestamp"] = datetime.now().isoformat()
    return state


# =====================================================================
# NODO 7: DESPLEGAR A PRODUCCIÓN
# =====================================================================

def deploy_to_production(state: MLOpsState) -> MLOpsState:
    """
    Node 7: Despliega el modelo aprobado a producción.
    
    Acción:
    - Mueve artefacto del modelo al bucket de producción
    - Actualiza registro de modelos
    - Dispara actualización del servicio de modelo
    - Actualiza production_metrics en estado
    
    Transición: Siempre termina el grafo (éxito)
    """
    logger.info("🚀 Node 7: Desplegando modelo a producción...")
    
    try:
        model_version = state.get("model_version", 1)
        candidate_path = state.get("candidate_model_path", "")
        
        # TODO: Implementar lógica real de despliegue
        # 1. Copiar modelo a bucket de producción
        # 2. Actualizar modelo registry
        # 3. Dispara actualización del servicio de modelo
        
        logger.info(f"✅ Modelo v{model_version} desplegado exitosamente!")
        logger.info(f"📦 Ruta del modelo: {candidate_path}")
        
        # Actualizar métricas de producción
        state["production_metrics"] = state.get("candidate_metrics", {})
        state["timestamp"] = datetime.now().isoformat()
        
        logger.info("✅ Métricas de producción actualizadas. ¡Pipeline completado!")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error desplegando a producción: {str(e)}")
        state["failure_analysis_report"] = f"Fallo en despliegue: {str(e)}"
        return state


# =====================================================================
# ESTADO INICIAL
# =====================================================================

def get_initial_state() -> MLOpsState:
    """Retorna un estado inicial limpio para el orquestador."""
    return {
        "new_data_path": "",
        "model_version": 1,
        "candidate_model_path": "",
        "candidate_metrics": {},
        "production_metrics": {},
        "deployment_decision": "",
        "failure_analysis_report": "",
        "data_quality_score": 0.0,
        "analysis_reason": "",
        "timestamp": ""
    }
