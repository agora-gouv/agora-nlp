import os
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from datetime import timedelta, datetime
from dags.config import DATA_FOLDER, FILEPATH, CLEANED_DATA_FILEPATH

from agora_topic_modeling.code.data_preparation import get_cleaned_doc_from_question

@dag(default_args={'owner': 'airflow'}, schedule=timedelta(minutes=2),
     start_date=pendulum.today('UTC').add(hours=-1))
def data_preparation():


    @task 
    def read_data(filepath: str)-> pd.DataFrame:
        data = pd.read_csv(filepath)
        return data


    @task
    def clean_data(data: pd.DataFrame, question_col: str, response_col: str, question: str):
        cleaned_data = get_cleaned_doc_from_question(data, question_col, response_col, question)
        cleaned_data.to_csv(CLEANED_DATA_FILEPATH)
        
    ## Fill stat dict ???

    question_col = ""
    response_col = ""
    question = ""
    data = read_data(FILEPATH)
    clean_data(data, question_col, response_col, question)

