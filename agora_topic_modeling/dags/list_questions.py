import os
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from datetime import timedelta
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from dags.config import DATABASE_CONFIG_FILEPATH



@dag(default_args={'owner': 'airflow'}, schedule=timedelta(days=30),
     start_date=pendulum.today('UTC').add(hours=-1))
def list_questions():

    @task 
    def get_list_questions()-> None:
        from code.psql_utils import get_connection_from_url
        load_dotenv()
        url = os.getenv("AGORA_PROD_URL")
        print(url)
        con = get_connection_from_url(url)
        df = pd.read_sql_query("SELECT question_id FROM reponses_consultation GROUP BY question_id ORDER BY question_id", con=con)
        print(df)
        con.close()
        return

    get_list_questions()

list_questions_dag = list_questions()
