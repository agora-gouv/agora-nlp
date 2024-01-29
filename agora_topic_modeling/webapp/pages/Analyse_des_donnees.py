import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os 
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from code.data_preparation import read_and_prep_data_from_question_id
from code.topic_modeling import get_topics_and_subtopics_from_df
from code.sentiment_analysis import apply_sentiment_analysis
from code.conversion_before_sql import prep_and_insert_to_agora_nlp
from code.psql_utils import get_connection_from_url

def data_selection():
    options = ["SQL", "Fichier"]
    choice = st.radio("Choisissez le type de données à analyser: ", options)
    return choice


def insert_question_id():
    question_id = st.text_input("Id de la question à analyser:")
    return question_id    

def get_list_questions()-> None:
    con = get_connection_from_url(os.environ.get("AGORA_PROD_URL"))
    df_question = pd.read_sql_query("SELECT * FROM questions WHERE type='open'", con)
    df_consult = pd.read_sql_query("SELECT id AS consultation_id, title AS consultation_title FROM consultations", con)
    df_question = df_question.merge(df_consult, on="consultation_id")
    st.write("Liste des questions ouvertes et leur ID")
    st.dataframe(df_question, use_container_width=True)
    con.close()
    return df_question



def write():
    df = None
    choice = data_selection()
    if choice == "SQL":
        df_question = get_list_questions()
        question_id = st.text_input("Id de la question à analyser:")
        if question_id != "":
            df = read_and_prep_data_from_question_id(question_id)
            

    if df is not None:
        row = df_question[df_question["id"] == question_id]
        question = row["title"].values[0]
        consultation_name = row["consultation_title"].values[0]
        st.write(f"Question: {question}")
        st.write(f"Consultation_title: {consultation_name}")
        display = st.checkbox("Afficher les réponses :")
        if display:
            st.dataframe(df, use_container_width=True)
        # Topic Modeling
        start_computing = st.checkbox("Lancer les calculs")
        if start_computing:
            with st.spinner("Caclul des topics et sous topics pour la question: "):

                doc_infos = get_topics_and_subtopics_from_df(df, "fracked_text")
            st.write("### Données après la modélisation des topics")
            st.dataframe(doc_infos)
            with st.spinner("Calcul de l'analyse de sentiment"):
                doc_infos_w_sentiments = apply_sentiment_analysis(doc_infos)
            st.write("### Données après l'analyse de sentiment")
            st.dataframe(doc_infos_w_sentiments)
            send_to_sql = st.button("Envoyer les données sur la base Agora NLP")
            if send_to_sql:
                prep_and_insert_to_agora_nlp(doc_infos_w_sentiments, question, consultation_name)
                return
    return



if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📊", page_title="Agora -- NLP"
    )
    write()
