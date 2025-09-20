from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import sys
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomExecption
from src.logger import logging
import os
import numpy as np
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    Preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.asyncdata_transformation_config=DataTransformationConfig()

    def get_trasformer_obj(self):
        try:
            num_col=['reading_score','writing_score']
            cat_col=['gender',
                     'race_ethnicity',
                     'parental_level_of_education',
                     'lunch',
                     'test_preparation_course']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())]
                    )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                            ('one_hot_encoder',OneHotEncoder()),
                            ('scaler',StandardScaler(with_mean=False))
                        ]
                    )
            logging.info("Numerical and Categorical columns are defined")
            preprocesor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_col),
                    ("cat_pipeline",cat_pipeline,cat_col)
                ]
                )
            return preprocesor
        except Exception as e:
            raise CustomExecption(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                logging.info("Read train and test data completed")
                preprocesor_obj=self.get_trasformer_obj()
                targer_col='math_score'
                num_col=['reading_score','writing_score']
                input_feature_train_df=train_df.drop(columns=[targer_col],axis=1)
                target_feature_train_df=train_df[targer_col]
                input_feature_test_df=test_df.drop(columns=[targer_col],axis=1)
                target_feature_test_df=test_df[targer_col]
                logging.info("Applying preprocessing object on training and testing datasets")
                input_feature_train_arr=preprocesor_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocesor_obj.transform(input_feature_test_df)
                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                logging.info("Saved preprocessing object")
                save_obj(

                file_path=self.asyncdata_transformation_config.Preprocessor_obj_file_path,
                obj=preprocesor_obj

            )
                return (
                    train_arr,
                    test_arr,
                    self.asyncdata_transformation_config.Preprocessor_obj_file_path
                )
            except Exception as e:
                raise CustomExecption(e,sys)   