import os
import re
import pandas as pd
import uuid
from code.psql_utils import get_connection_from_url

def fracking(df: pd.DataFrame, col: str, sep: str)-> pd.DataFrame:
    # lambda split that filter empty elements from list
    custom_filter = lambda x: x not in ["", " ", None]
    custom_split = lambda x: list(filter(custom_filter, x.split(sep)))

    df = df.copy()
    df["fracked_text"] = df[col].apply(custom_split)
    df = df.explode("fracked_text")
    df["old_index"] = df.index
    fracking_count = df.groupby("old_index").agg(fracking_count=("old_index", "count")).reset_index()
    df = df.merge(fracking_count, on="old_index")
    return df


def compute_response_size(df: pd.DataFrame, response_col: str)-> pd.DataFrame:
    df_ = df.copy()
    df_["response_size"] = df_[response_col].str.split().apply(len)
    return df_


# Remove, useless answers
def prep_answer_df(df: pd.DataFrame, response_col: str):
    cleaned_df = df.dropna(subset=response_col).copy()
    NUMBERED_LIST_HANDLER = lambda x: re.sub(r'([0-9]+\. )', ' ', x)
    SPECIAL_CHAR_HANDLER = lambda x: re.sub(r'[()/\\]', ' ', x)
    WHITESPACE_HANDLER = lambda x: re.sub('\s+', ' ', re.sub('\n+', ' ', x.strip()))
    cleaned_df["cleaned_text"] = cleaned_df[response_col].apply(NUMBERED_LIST_HANDLER)
    cleaned_df["cleaned_text"] = cleaned_df["cleaned_text"].apply(SPECIAL_CHAR_HANDLER)
    cleaned_df["cleaned_text"] = cleaned_df["cleaned_text"].apply(WHITESPACE_HANDLER)
    fracked_df = fracking(cleaned_df, "cleaned_text", sep=".")
    fracked_df = fracked_df.dropna(axis=0, subset="fracked_text").reset_index(drop=True)
    fracked_df = compute_response_size(fracked_df, "fracked_text")
    return fracked_df


def get_cleaned_doc_from_question(df: pd.DataFrame, question_col: str, response_col: str, question: str)-> pd.DataFrame:
    df_filtered = df[df[question_col] == question].copy()
    cleaned_doc = prep_answer_df(df_filtered, response_col)
    return cleaned_doc


def gen_uuid()-> str:
    myuuid = uuid.uuid4()
    return str(myuuid)


def gen_uuid_col(size: int):
    col = [gen_uuid() for i in range(size)]
    return col


def prep_before_sql(doc_infos: pd.DataFrame)-> pd.DataFrame:
    preped_df = doc_infos.copy()
    columns = {"Document": "text",
               "Probability": "topic_probability",
               "Topic": "topic",
               "old_index": "origin_response_id"}
    preped_df = preped_df.rename(columns=columns)
    preped_df.loc[:, "subtopic_id"] = preped_df.groupby(["topic", "sub_topic"]).topic.transform(lambda g: uuid.uuid4())
    preped_df.loc[:, "topic_id"] = preped_df.groupby("topic").topic.transform(lambda g: uuid.uuid4())
    columns_to_keep = ["text", "topic_id", "subtopic_id", "topic_probability", "sentiment", "sentiment_score", "origin_response_id", "Name", "sub_name"]
    return preped_df[columns_to_keep]


def prep_topic(preped_df: pd.DataFrame)-> pd.DataFrame:
    topics = preped_df.groupby("topic_id").agg(name=("Name", "first")).reset_index()
    return topics


def prep_sub_topics(preped_df: pd.DataFrame):
    sub_topics = preped_df.groupby("subtopic_id").agg(name=("sub_name", "first"), parent_id=("topic_id", "first")).reset_index()
    sub_topics = sub_topics.rename(columns={"subtopic_id": "topic_id"})
    return sub_topics


# Main function for data preparation
def read_and_prep_data_from_question_id(question_id: str)-> pd.DataFrame:
    url = os.getenv("AGORA_PROD_URL")
    print("URL")
    print(url)
    con = get_connection_from_url(url)
    query = f"SELECT * FROM reponses_consultation WHERE question_id='{question_id}'"
    df = pd.read_sql_query(query, con=con)
    cleaned_data = prep_answer_df(df, "response_text")
    return cleaned_data
