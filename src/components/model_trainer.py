import os
import sys
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.exception import CustomExecption
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj,evl_model

@dataclass
class ModelTrainerConfig:
    train_model_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestClassifier(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostClassifier()
            }
            
            model_report = evl_model(X_train, y_train, X_test, y_test, models)
            ## To get the best model score from dict
            best_model_score=max(sorted(model_report.values()))
            ## To get the best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomExecption("No best model found", sys)
            print(f"Best model found ,model name is {best_model_name} and score is {best_model_score}")
            logging.info(f"Best model found ,model name is {best_model_name} and score is {best_model_score}")
            save_obj(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomExecption(e, sys)
            
    