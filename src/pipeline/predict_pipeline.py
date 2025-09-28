import sys
import pandas as pd
from src.utils import load_object
from src.exception import CustomExecption

class PredictionPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,feature):
        try:
            model_path='artifacts/model.pkl'
            prepocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            prepocessor=load_object(file_path=prepocessor_path)
            data_scaled=prepocessor.transform(feature)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomExecption(e,sys)
        
        
class Customdata():
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,reading_score:int,writing_score:int): 
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def get_data_as_dataframe(self):
        try:
            custom_data_input={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
                }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise Exception(e,sys)
            
        