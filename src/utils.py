import os
import pandas as pd
import numpy as np
import sys
import pickle
from src.exception import CustomExecption
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomExecption(e,sys)
    
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold
from collections import Counter

def evl_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomExecption(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomExecption(e,sys)