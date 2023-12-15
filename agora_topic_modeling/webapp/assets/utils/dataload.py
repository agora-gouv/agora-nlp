import pandas as pd
import streamlit as st
from bertopic import BERTopic
import json


@st.cache_data
def load_model(model_path: str)-> BERTopic:
    custom_bertopic = BERTopic.load(model_path)
    return custom_bertopic


@st.cache_data
def load_cleaned_labels(question_name: str, folder: str)-> list[list[str]]:
    with open(folder + question_name + "/cleaned_labels.json") as f:
        data = json.load(f)
    return data


@st.cache_data
def load_doc_infos(filepath: str) -> pd.DataFrame:
    doc_infos = pd.read_csv(filepath, index_col=0)
    doc_infos["Topic"] = doc_infos["Topic"].astype(int)
    doc_infos = doc_infos.sort_values("Probability", ascending=False)
    return doc_infos
