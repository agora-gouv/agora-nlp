import os
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from datetime import timedelta

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dags.config import DATABASE_CONFIG_FILEPATH
from code.psql_utils import get_connection


@dag(default_args={'owner': 'airflow'}, schedule=timedelta(days=1),
     start_date=pendulum.today('UTC').add(hours=-1))
def list_questions():

    @task 
    def get_list_questions()-> None:
        con = get_connection(DATABASE_CONFIG_FILEPATH, "local_prod")
        df = pd.read_sql_query("SELECT id, title FROM questions WHERE type='open'", con=con)
        print(df)
        con.close()
        return

    get_list_questions()

list_questions_dag = list_questions()