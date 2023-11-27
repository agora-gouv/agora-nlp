import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic

from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
print(path_root)
sys.path.append(str(path_root))

from agora_topic_modeling.code.topic_dataviz import create_wordcloud_from_topic

TOPIC_FOLDER = "data/topic_modeling/"

def plot_mattrix(data):
    fig, ax = plt.subplots(figsize = (2, 2))
    plt.imshow(data, interpolation='nearest')
    plt.xlabel("Label")
    plt.ylabel("Label")
    plt.title("Cosine Similarity")
    st.pyplot(fig)


@st.cache_data()
def measure_similarity_of_topic(topic_labels: list[str], _topic_model):
    embedding = _topic_model.embedding_model.embed(topic_labels)
    similarity_matrix = cosine_similarity(embedding)

    top_id = np.argmax(np.sum(similarity_matrix, axis=1))

    top_label = topic_labels[top_id]

    # scoring topic
    triu_mat = np.triu(similarity_matrix, k=1)
    score = np.mean(triu_mat[np.nonzero(triu_mat)])

    return similarity_matrix, top_label, score


@st.cache_data()
def load_model(model_path: str)-> BERTopic:
    custom_bertopic = BERTopic.load(model_path)
    return custom_bertopic

@st.cache_data()
def load_doc_infos(filepath: str) -> pd.DataFrame:
    doc_infos = pd.read_csv(filepath)
    doc_infos["Topic"] = doc_infos["Topic"].astype(int)
    st.write(doc_infos.info())
    return doc_infos


def get_doc_stats(doc_infos: pd.DataFrame)-> pd.DataFrame:
    doc_infos_upgraded = doc_infos.copy()
    threshold = 0.8
    doc_infos_upgraded["Good_proba"] = doc_infos_upgraded["Probability"] >= threshold
    stats = doc_infos_upgraded.groupby("Topic").agg(nb_doc=("Document", "count"), good_docs=("Good_proba", sum))
    return stats


def topic_selection(custom_bertopic: BERTopic, stats: pd.DataFrame, cleaned_labels: list[list[str]], question_short: str):
    topic_count = 8
    wc_tab, label_tab = st.tabs(["Wordcloud", "Labeling"])
    st.markdown("---")
    with wc_tab:
        force_compute = st.button("Recalculer les nuages de mots")
        wc_folder = TOPIC_FOLDER + question_short + "/wordcloud/"
        st.write("Computing Wordclouds")
        wc_columns = st.columns(4)
        for i in range(topic_count):
            wc_filepath = wc_folder + "wc_" + str(i) + ".png"
            # Si les nuages de mots n'existent pas les calculer
            if not os.path.isdir(wc_folder) or force_compute:
                os.makedirs(wc_folder, exist_ok=True) 
                wordcloud = create_wordcloud_from_topic(custom_bertopic, i)
                wordcloud.to_file(wc_filepath)
            # Afficher les nuages de mots
            with wc_columns[i%4]:
                st.write("### Topic " + str(i))
                st.image(wc_filepath, width=300)
                                
    with label_tab:
        topic = st.selectbox("Sélectionnez le topic à observer : ", range(len(cleaned_labels)))
        if topic is not None:
            st.write("### Topic " + str(topic))
            label_cols = st.columns(2)
            
            
            sim_matrix, top_label, score = measure_similarity_of_topic(cleaned_labels[topic], custom_bertopic)
            with label_cols[0]:
                st.write("#### Infos sur le topic")
                nb_doc = stats.loc[topic]["nb_doc"]
                st.write("Nombre de documents : " + str(nb_doc))
                st.write("Meilleur label généré d'après le modèle : ")
                st.write("**" + top_label + "**")
                st.write("*Score de confiance : " + str(score) + "*")

                other_labels = st.checkbox("Afficher les labels potentiels ?")
                if other_labels:
                    st.write("Les labels potentiels sont : ")
                    for i in range(len(cleaned_labels[topic])):
                        st.write("*" + cleaned_labels[topic][i] + "*")
            with label_cols[1]:
                st.write("#### Matrices de similarité entre les labels")
                plot_mattrix(sim_matrix)
                st.write(sim_matrix)


def load_cleaned_labels(question_name: str)-> list[list[str]]:
    with open(TOPIC_FOLDER+question_name + "/cleaned_labels.json") as f:
        data = json.load(f)
    return data


def write():
    st.write("# Question : Avez-vous des propositions pour financer la transition écologique ? C’est la dernière question !")
    st.write("## Evaluation des topics générés")
    question_short = st.selectbox("Choisissez la question à analyser :", options=["question_1", "transition_ecologique"])
    #TODO: Select question then load model and filepath depending on choice
    model_path = "data/topic_modeling/" + question_short + "/bertopic_model"
    custom_bertopic = load_model(model_path)
    fig = custom_bertopic.visualize_barchart()
    st.plotly_chart(fig)
    filepath = "data/topic_modeling/" + question_short + "/doc_infos.csv"
    doc_infos = load_doc_infos(filepath)
    stats = get_doc_stats(doc_infos)
    
    
    cleaned_labels = load_cleaned_labels(question_short)
    topic_selection(custom_bertopic, stats, cleaned_labels, question_short)
    return


if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📊", page_title="Agora -- NLP"
    )
    write()
