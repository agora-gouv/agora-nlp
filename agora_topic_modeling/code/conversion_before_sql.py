import pandas as pd
import os
import uuid
from sqlalchemy import create_engine, select, inspect
from sqlalchemy.engine import URL
from code.psql_utils import get_connection_from_url

def prep_before_sql(doc_infos: pd.DataFrame)-> pd.DataFrame:
    preped_df = doc_infos.copy()
    columns = {"Document": "text",
               "Probability": "topic_probability",
               "Topic": "topic"}
    preped_df = preped_df.rename(columns=columns)
    preped_df.loc[:, "sub_topic_id"] = preped_df.groupby(["topic", "sub_topic"]).topic.transform(lambda g: uuid.uuid4())
    preped_df.loc[:, "topic_id"] = preped_df.groupby("topic").topic.transform(lambda g: uuid.uuid4())
    
    #Here the column taken before apply ["topic"] doesn't matter, it is just to generate a random uuid for each row
    preped_df.loc[:, "id"] = preped_df["topic"].apply(lambda g: uuid.uuid4())
    
    origin_response = preped_df.groupby("old_index").agg(origin_response_id=("id", "first"))
    preped_df = preped_df.join(origin_response, on="old_index")
    #preped_df["fracking_count"] = preped_df["fracking_count"].astype(int)

    columns_to_keep = ["id", "text", "topic_id", "sub_topic_id", "origin_response_id", "topic_probability", "sentiment", "sentiment_score", "Name", "sub_name", "fracking_count", "Representative_document"]
    return preped_df[columns_to_keep]


def prep_sub_topics(preped_df: pd.DataFrame):
    sub_topics = preped_df.groupby("sub_topic_id").agg(name=("sub_name", "first"), parent_topic_id=("topic_id", "first")).reset_index()
    sub_topics = sub_topics.rename(columns={"sub_topic_id": "id"})
    return sub_topics


def prep_topic(preped_df: pd.DataFrame, question: str, consultation_name: str)-> pd.DataFrame:
    topics = preped_df.groupby("topic_id").agg(name=("Name", "first")).reset_index()
    topics = topics.rename(columns={"topic_id": "id"})
    sub_topics = prep_sub_topics(preped_df)
    
    consultation_id = uuid.uuid4()
    consultation_df = pd.DataFrame({"title": [consultation_name], "id":[consultation_id]})
    question_id = uuid.uuid4()
    question_df = pd.DataFrame({"title": [question], "id":[question_id], "consultation_id": [consultation_id]})
    topics_df = pd.concat([topics, sub_topics])
    topics_df["question_id"] = question_id
    return consultation_df, question_df, topics_df


def prep_representative_answers(preped_df: pd.DataFrame):
    representative_df = preped_df[preped_df["Representative_document"]][["topic_id", "id"]]
    representative_df = representative_df.rename(columns={"id": "response_id"})
    return representative_df


def get_engine(section="postgresql"):
    url = os.getenv("AGORA_NLP_URL")
    engine = create_engine(url)
    return engine



def prep_and_insert_to_agora_nlp(doc_infos: pd.DataFrame, question: str, consultation_name: str):
    preped_df = prep_before_sql(doc_infos)
    representative_df = prep_representative_answers(preped_df)
    consultation_df, question_df, topics_df = prep_topic(preped_df, question, consultation_name)
    responses_df = preped_df.drop(columns=["Name", "sub_name", "Representative_document"])
    if_exists="append"
    url = os.get_env("AGORA_NLP_URL")
    with get_connection_from_url(url) as con:
        consultation_df.to_sql("consultations", con, if_exists=if_exists, index=False)
        question_df.to_sql("questions", con, if_exists=if_exists, index=False)
        topics_df.to_sql("topics", con, if_exists=if_exists, index=False)
        responses_df.to_sql("responses", con, if_exists=if_exists, index=False)
        representative_df.to_sql("representative_responses", con, if_exists=if_exists, index=False)
    return
