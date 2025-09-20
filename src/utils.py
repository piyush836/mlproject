import os
import pandas as pd
import numpy as np
import sys
import pickle
from src.exception import CustomExecption

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomExecption(e,sys)
