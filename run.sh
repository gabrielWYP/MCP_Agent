#!/bin/bash
# Script para ejecutar el proyecto MLOps desde terminal
# Uso: ./run.sh [comando] [opciones]

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directorio del proyecto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Verificar Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 no está instalado${NC}"
    exit 1
fi

# Mostrar versión de Python
PYTHON_VERSION=$(python3 --version)
echo -e "${BLUE}Usando: $PYTHON_VERSION${NC}"

# Función para mostrar ayuda
show_help() {
    cat << EOF
${BLUE}MLOps Orchestrator - Mango Ripeness Detection Pipeline${NC}

${GREEN}Uso: ./run.sh [comando] [opciones]${NC}

${YELLOW}Comandos:${NC}
  mlops          Ejecutar orquestador MLOps principal (default)
  homography     Ejecutar calibración de homografía RGB-NIR
  help           Mostrar esta ayuda

${YELLOW}Ejemplos:${NC}
  # Ejecutar orquestador MLOps
  ./run.sh
  ./run.sh mlops

  # Ejecutar calibración de homografía con parámetros por defecto
  ./run.sh homography

  # Ejecutar calibración con directorios personalizados
  ./run.sh homography --dir-rgb "calibracion/rgb/*/*.jpg" --dir-nir "calibracion/nir/*/*.jpg"

  # Ejecutar calibración con tablero Charuco personalizado
  ./run.sh homography --squares-x 12 --squares-y 9 --square-length 0.025

  # Ejecutar con logging detallado
  ./run.sh homography --verbose

  # Ver todas las opciones disponibles
  python3 src/main.py -h
  python3 src/main.py homography -h

${YELLOW}Variables de entorno:${NC}
  PYTHONPATH      Se establece automáticamente a \${PROJECT_DIR}
  CONFIG_FILE     Ruta al archivo .env (se carga automáticamente)

EOF
}

# Función para ejecutar MLOps
run_mlops() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}INICIANDO ORQUESTADOR MLOPS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" python3 src/main.py mlops "$@"
}

# Función para ejecutar homografía
run_homography() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}INICIANDO CALIBRACIÓN DE HOMOGRAFÍA${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" python3 src/main.py homography "$@"
}

# Procesar argumentos
CMD="${1:-mlops}"
shift || true

case "$CMD" in
    mlops)
        run_mlops "$@"
        ;;
    homography)
        run_homography "$@"
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo -e "${RED}Error: Comando desconocido '$CMD'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Ejecución completada exitosamente${NC}"
else
    echo -e "${RED}❌ Error durante ejecución (código: $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE
