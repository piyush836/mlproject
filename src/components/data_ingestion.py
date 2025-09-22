import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from src.logger import logging
from src.exception import CustomExecption
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class Dataingestionconfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class Dataingestion:
    def __init__(self):
        self.config=Dataingestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df=pd.read_csv('Notebook\Data\stud.csv')
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path,index=False)
            logging.info("Train test split intiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.config.train_data_path,index=False,header=True)
            test_set.to_csv(self.config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            return(
                self.config.train_data_path,
                self.config.test_data_path)
        except Exception as e:
            raise CustomExecption(e,sys)

if __name__=="__main__":
    obj=Dataingestion()
    obj.initiate_data_ingestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array) )   