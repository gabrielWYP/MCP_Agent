from storage_logic.logic import check_new_objects
from utils.logger import logger_singleton as logger
from variables.configvariables import BUCKET_NAME

def process():
    print('BUCKET_NAME:', BUCKET_NAME)
    have_news = check_new_objects(BUCKET_NAME)
    if have_news:
        logger.warning(
            "New data found, but the retired training entrypoint was removed. "
            "Use scripts/download_oci.py, Label Studio export, "
            "scripts/convert_nir_labels.py, and python -m src.training.train."
        )
    else:
        logger.info("No new data found. Skipping training pipeline.")
