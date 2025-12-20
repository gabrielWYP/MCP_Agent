import json
import os
from storage_logic.logic import check_new_objects
from utils.logger import logger_singleton as logger
from ml_pipeline.training import training_process
from variables.configvariables import BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT_URL

def process():
    print('BUCKET_NAME:', BUCKET_NAME)
    have_news = check_new_objects(BUCKET_NAME)
    if have_news:
        training_process()
    else:
        logger.info("No new data found. Skipping training pipeline.")