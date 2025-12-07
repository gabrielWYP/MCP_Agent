import json
import os
from storage_logic.logic import check_new_objects
from variables.configvariables import BUCKET_NAME, S3_BUCKET_NAME, S3_ACCESS_KEY, S3_SECRET_KEY, S3_ENDPOINT_URL

def process():
    check_new_objects(BUCKET_NAME)