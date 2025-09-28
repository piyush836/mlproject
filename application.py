from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata,PredictionPipeline
application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def perform_prediction():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            
           
            gender=request.form.get('gender', ''),
            race_ethnicity=request.form.get('ethnicity', ''),
            parental_level_of_education=request.form.get('parental_level_of_education', ''),
            lunch=request.form.get("lunch", ""),
            test_preparation_course=request.form.get("test_preparation_course", ""),
            reading_score=int(request.form.get("reading_score", 0)),
            writing_score=int(request.form.get("writing_score", 0))
)


        
        predict_df=data.get_data_as_dataframe()
        print(predict_df)
        Prdiction=PredictionPipeline()
        results=Prdiction.predict(predict_df)
        return render_template('home.html',results=results)

if __name__=="__main__":
   
    app.run(host='127.0.0.1')
 
    
        