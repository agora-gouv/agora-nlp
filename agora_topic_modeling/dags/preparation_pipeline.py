import os
import pendulum

from airflow.decorators import dag, task
from airflow.models.param import Param
from datetime import timedelta, datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dags.config import DATA_FOLDER, DATABASE_CONFIG_FILEPATH, CLEANED_DATA_FILEPATH


@dag(default_args={'owner': 'airflow'}, params={"question_id": "Replace with question_id"}, schedule=timedelta(days=30),
     start_date=pendulum.today('UTC').add(hours=-1))
def data_preparation_pipeline():


    @task 
    def read_data(**context)-> str:
        from code.psql_utils import get_connection_from_url
        import pandas as pd
        question_id = context["params"]["question_id"]
        tmp_path = os.path.join(DATA_FOLDER, 'data_tmp.parquet')
        url = os.getenv("AGORA_PROD_URL")
        con = get_connection_from_url(url)
        query = f"SELECT * FROM reponses_consultation WHERE question_id='{question_id}'"
        df = pd.read_sql_query(query, con=con)
        df.to_parquet(tmp_path)
        return tmp_path


    @task
    def clean_data(tmp_path: str):
        from code.data_preparation import prep_answer_df
        import pandas as pd
        data = pd.read_parquet(tmp_path)
        cleaned_data = prep_answer_df(data, "response_text")
        cleaned_data.to_parquet(CLEANED_DATA_FILEPATH)
        return

    
    tmp_path = read_data()
    clean_data(tmp_path)

data_preparation_dag = data_preparation_pipeline()