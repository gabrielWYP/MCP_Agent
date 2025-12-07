import os
from variables.configvariables import BUCKET_NAME
from utils.ObjectStorage import count_and_list_objects

def check_new_objects(BUCKET_NAME: str) -> bool:
    full_path = BUCKET_NAME + '/news'
    count, objects = count_and_list_objects(full_path)
    
    if count > 0:
        return True
    else:
        return False