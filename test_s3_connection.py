import os
import sys
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env en la raíz del proyecto
load_dotenv()

# Añadimos el directorio de notebooks al path para poder importar utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'notebooks'))

from notebooks.utils import list_all_bucket_objects, BUCKET_NAME

if __name__ == "__main__":
    print("\n--- Iniciando prueba de conexión S3 a OCI ---")
    try:
        objects = list_all_bucket_objects(BUCKET_NAME)
        if objects:
            print(f"Objetos encontrados en el bucket '{BUCKET_NAME}':")
            for obj in objects:
                print(f" - {obj}")
        else:
            print(f"El bucket '{BUCKET_NAME}' está vacío o no se encontraron objetos.")
    except ValueError as ve:
        print(f"Error de configuración: {ve}")
        print("Asegúrate de que las variables de entorno S3_ENDPOINT_URL, S3_ACCESS_KEY y S3_SECRET_KEY estén configuradas.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
    print("\n--- Prueba finalizada ---")
