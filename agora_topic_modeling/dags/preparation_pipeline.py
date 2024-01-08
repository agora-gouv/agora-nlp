import os
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.models.param import Param
from datetime import timedelta, datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dags.config import DATA_FOLDER, DATABASE_CONFIG_FILEPATH, CLEANED_DATA_FILEPATH

from code.psql_utils import get_connection
from code.data_preparation import prep_answer_df


@dag(default_args={'owner': 'airflow'}, params={"question_id": "Replace with question_id"}, schedule=timedelta(days=30),
     start_date=pendulum.today('UTC').add(hours=-1))
def data_preparation():


    @task 
    def read_data(**context)-> pd.DataFrame:
        question_id = context["params"]["question_id"]
        #features_path = os.path.join(DATA_FOLDER, f'cleaned_data_{datetime.now()}.parquet')
        tmp_path = os.path.join(DATA_FOLDER, f'data_tmp.parquet')
        con = get_connection(DATABASE_CONFIG_FILEPATH, "local_prod")
        query = f"SELECT * FROM reponses_consultation WHERE question_id='{question_id}'"
        df = pd.read_sql_query(query, con=con)
        df.to_parquet(tmp_path)
        return tmp_path


    @task
    def clean_data(tmp_path: str):
        data = pd.read_parquet(tmp_path)
        cleaned_data = prep_answer_df(data, "response_text")
        print(cleaned_data)
        cleaned_data.to_parquet(CLEANED_DATA_FILEPATH)
        return

    
    #question_id = "5563aeda-092a-11ee-be56-0242ac120002"
    tmp_path = read_data()
    clean_data(tmp_path)

data_preparation_dag = data_preparation()