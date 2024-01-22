import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import pendulum


import pandas as pd
import pandas as pd

from airflow.decorators import dag, task
from datetime import timedelta

import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from dags.config import DOC_INFOS_FILEPATH, DOC_INFOS_WITH_SUB_FILEPATH, CLEANED_DATA_FILEPATH, TOPIC_THRESHOLD_FOR_SUBTOPIC
from code.topic_modeling import get_custom_bertopic_model, get_topic_distribution, get_sub_topics



@dag(default_args={'owner': 'airflow'}, schedule=timedelta(days=30),
     start_date=pendulum.today('UTC').add(hours=-1))
def topic_modeling_pipeline_2():


    @task
    def topic_modeling_2():
        
    
        cleaned_data = pd.read_parquet(CLEANED_DATA_FILEPATH)
        fracking_merge = cleaned_data[["old_index", "fracking_count"]].copy()
        X = cleaned_data["fracked_text"].head(10)
        print("Begin Fit BerTopic Model")
        print(X.shape)
        custom_bert, custom_topics = get_custom_bertopic_model(X)
        print("Get document info from BerTopic")
        doc_infos = custom_bert.get_document_info(X)
        doc_infos = doc_infos.join(fracking_merge)
        doc_infos.to_parquet(DOC_INFOS_FILEPATH)
        return DOC_INFOS_FILEPATH

    
    @task
    def subtopic_modeling_2(filepath: str):
        doc_infos = pd.read_parquet(filepath)
        answers_per_topic = get_topic_distribution(doc_infos)
        sub_topic_range = answers_per_topic[answers_per_topic["percentage"] > TOPIC_THRESHOLD_FOR_SUBTOPIC].count()[0]
        doc_infos_w_subtopics = get_sub_topics(doc_infos, sub_topic_range)
        doc_infos_w_subtopics.to_parquet(DOC_INFOS_WITH_SUB_FILEPATH)

    
    filepath = topic_modeling_2()
    subtopic_modeling_2(filepath)

topic_modeling_dag = topic_modeling_pipeline_2()