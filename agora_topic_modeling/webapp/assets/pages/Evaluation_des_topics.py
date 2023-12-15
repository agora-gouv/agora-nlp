import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import shlex
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
import collections
from nltk.corpus import stopwords

from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
print(path_root)
sys.path.append(str(path_root))
from agora_topic_modeling.code.topic_dataviz import create_wordcloud_from_topic
from assets.utils.dataload import load_model, load_cleaned_labels, load_doc_infos
from assets.utils.datafilter import get_sentences_with_words, get_sentences_including_words
#from assets.utils.dataviz import most_common_2g

TOPIC_FOLDER = "data/topic_modeling/"


@st.cache_data
def measure_similarity_of_topic(topic_labels: list[str], _topic_model):
    embedding = _topic_model.embedding_model.embed(topic_labels)
    similarity_matrix = cosine_similarity(embedding)
    top_id = np.argmax(np.sum(similarity_matrix, axis=1))
    top_label = topic_labels[top_id]

    # scoring topic
    triu_mat = np.triu(similarity_matrix, k=1)
    score = np.mean(triu_mat[np.nonzero(triu_mat)])
    return similarity_matrix, top_label, score


@st.cache_data
def remove_stopwords(sentence: str, stopwords: list[str])-> list[str]:
    tokens = sentence.lower().split(" ")
    result = []
    for token in tokens:
        if token not in stopwords:
            result.append(token)
    return result


@st.cache_data
def get_tokens_without_stopwords(df: pd.DataFrame, col: str)-> pd.DataFrame:
    stop_words=stopwords.words("french")
    df["tokens"] = df["Document"].apply(lambda x: remove_stopwords(x, stop_words))
    return df


@st.cache_data
def prep_doc_info(doc_infos: pd.DataFrame):
    doc_infos_prepped = doc_infos.copy()
    doc_infos_prepped["Answer_with_proba"] = "(" + doc_infos_prepped["Probability"].astype(str) + ") " + doc_infos_prepped["Document"] 
    doc_infos_prepped = get_tokens_without_stopwords(doc_infos_prepped, "Document")
    return doc_infos_prepped


@st.cache_data
def get_doc_stats(doc_infos: pd.DataFrame)-> pd.DataFrame:
    doc_infos_upgraded = doc_infos.copy()
    doc_count = len(doc_infos_upgraded.index)
    threshold = 0.8
    doc_infos_upgraded["Good_proba"] = doc_infos_upgraded["Probability"] >= threshold
    stats = doc_infos_upgraded.groupby("Topic").agg(nb_doc=("Document", "count"), good_docs=("Good_proba", sum))
    stats["percentage"] = (stats["nb_doc"] / doc_count) * 100
    stats["percentage"] = stats["percentage"].round(decimals=2)
    stats["nb_doc"] = stats["nb_doc"].astype(int)
    return stats


@st.cache_resource
def plot_frequent_words(freq_words: pd.DataFrame, title="Fréquence des mots du topic"):
    color_sequence = px.colors.sequential.Viridis.copy()
    color_sequence.reverse()
    fig = px.bar(freq_words, x="freq", y="word", color="word", title=title, orientation='h', color_discrete_sequence=color_sequence)
    fig.update_layout(showlegend=False, title_x=0.3)
    fig.update_xaxes(title="Fréquence des mots")
    fig.update_yaxes(title="Mots importants")
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def plot_mattrix(data):
    fig, ax = plt.subplots(figsize = (2, 2))
    plt.imshow(data, interpolation='nearest')
    plt.xlabel("Label")
    plt.ylabel("Label")
    plt.title("Cosine Similarity")
    st.pyplot(fig)


def display_outliers_topic(doc_infos: pd.DataFrame, custom_bertopic: BERTopic, stats: pd.DataFrame, is_subtopic: bool=False):
    type = "Sous-topic" if is_subtopic else "Topic"
    st.write(f"#### Exploration du {type} des cas particuliers pour la question")
    title = f"{type} -1 : {int(stats.loc[-1]['nb_doc'])} réponses ({stats.loc[-1]['percentage']}%)"
    freq_words = pd.DataFrame(custom_bertopic.get_topic(-1), columns=["word", "freq"])
    plot_frequent_words(freq_words, title)
    display_answers_from_topic(doc_infos, -1)


def display_topic_overview(custom_bertopic: BERTopic, stats: pd.DataFrame, is_subtopic: bool=False):
    type = "Sous-topic" if is_subtopic else "Topic"
    st.write(f"#### Vue d'ensemble des {type}s générés pour la question")
    topic_range = min(len(stats.index)-1, 6)
    cols = st.columns(3)
    for topic in range(topic_range):
        with cols[topic%3]:
            title = f"{type} {topic} : {int(stats.loc[topic]['nb_doc'])} réponses ({stats.loc[topic]['percentage']}%)"
            freq_words = pd.DataFrame(custom_bertopic.get_topic(topic), columns=["word", "freq"])
            plot_frequent_words(freq_words, title)
    return


def display_answers_from_topic(doc_info: pd.DataFrame, topic: int):
    exp = st.expander("Afficher la liste les réponses du topic sélectionné")
    with exp:
        st.dataframe(doc_info[doc_info["Topic"] == topic]["Answer_with_proba"].values, use_container_width=True)
    return


@st.cache_data
def get_most_present_words_g(df: pd.DataFrame, col: str, ngram: int):
    c=collections.Counter()
    for x in df[col]:
        #x = i.rstrip().split(" ")
        c.update(set(zip(x[:-1],x[1:])))
    most_presents_bigram = list(map(lambda x: (" ".join(x[0]), x[1]), c.most_common(10)))
    most_presents_bigram = pd.DataFrame(most_presents_bigram, columns=["bigram", "count"])
    st.write("##### Bi-gram les plus présents dans le topic :")
    st.write(most_presents_bigram)
    return most_presents_bigram


def subtopics_info(question_short: str, topic: str):
    subtopic_model_path = f"data/topic_modeling/{question_short}/bertopic_model_{topic}"
    if os.path.isdir(subtopic_model_path):
        subtopic_filepath = f"data/topic_modeling/{question_short}/doc_infos_{topic}.csv"
        st.write("#### Info sur les sous-topics")
        sub_bertopic = load_model(subtopic_model_path)
        sub_doc_infos = load_doc_infos(subtopic_filepath)
        sub_stats = get_doc_stats(sub_doc_infos)
        display_topic_overview(sub_bertopic, sub_stats, True)


def display_topic_basic_info(topic: int, cleaned_labels: pd.DataFrame, custom_bertopic: BERTopic, stats: pd.DataFrame):
    sim_matrix, top_label, score = measure_similarity_of_topic(cleaned_labels[topic], custom_bertopic)
    st.write("#### Infos sur le topic")
    nb_doc = stats.loc[topic]["nb_doc"]
    percentage = stats.loc[topic]["percentage"]
    st.write(f"Nombre de documents : **{int(nb_doc)}** *({str(percentage)}%)*")
    st.write("Meilleur label généré d'après le modèle : ")
    st.write("**" + top_label + "**")
    st.write("*Score de confiance : " + str(score) + "*")

    other_labels = st.checkbox("Afficher les labels potentiels ?")
    if other_labels:
        st.write("Les labels potentiels sont : ")
        for i in range(len(cleaned_labels[topic])):
            st.write("*" + cleaned_labels[topic][i] + "*")
    freq_words = pd.DataFrame(custom_bertopic.get_topic(topic), columns=["word", "freq"])
    plot_frequent_words(freq_words)


def display_topic_info(topic: int, doc_infos: pd.DataFrame, cleaned_labels: list[list[str]], custom_bertopic: BERTopic, question_short: str):
    stats = get_doc_stats(doc_infos)
    if topic is not None:
        topic_info = doc_infos[doc_infos["Topic"] == topic].copy()
        st.write("### Topic " + str(topic))
        label_cols = st.columns(2)
        with label_cols[0]:
            display_topic_basic_info(topic, cleaned_labels, custom_bertopic, stats)
            
        with label_cols[1]:
            wc_folder = TOPIC_FOLDER + question_short + "/wordcloud/"
            wc_filepath = wc_folder + "wc_" + str(topic) + ".png"
            st.image(wc_filepath, use_column_width=True)
            # topic_df = doc_infos[doc_infos["Topic"] == topic]
            # words_2g = most_common_2g(topic_df, "Document")
            # st.write(words_2g)
            #plot_mattrix(sim_matrix)
            most_presents_bigram = get_most_present_words_g(topic_info, "tokens", 2)
        
        selected_bigram = st.selectbox("Selectionner le bigram dont vous voulez voir les contributions", most_presents_bigram["bigram"].values)
        sentences = get_sentences_with_words(topic_info, "tokens", selected_bigram, "Document")
        #sentences = get_sentences_including_words(topic_info, "tokens", selected_bigram.split(), "Document")
        with st.expander("Réponses avec le bigram sélectionné"):
            st.dataframe(sentences, use_container_width=True)



        best = doc_infos[doc_infos["Topic"] == topic]["Representative_Docs"].values[0]
        best_answers = shlex.split(best[1:-1])
        expander = st.expander("Afficher les réponses pertinentes")
        for i in range(len(best_answers)):
            expander.write(best_answers[i])
        display_answers_from_topic(doc_infos, topic)
        # if folder exists
        subtopics_info(question_short, topic)


def topic_selection(custom_bertopic: BERTopic, doc_infos: pd.DataFrame, cleaned_labels: list[list[str]], question_short: str):
    topic_count = 8
    label_tab, wc_tab, outlier_tab = st.tabs(["Détails des Topics", "Nuages de mots", "Cas Particuliers"])
    st.markdown("---")
    with wc_tab:
        force_compute = st.button("Recalculer les nuages de mots")
        wc_folder = TOPIC_FOLDER + question_short + "/wordcloud/"
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
        display_topic_info(topic, doc_infos, cleaned_labels, custom_bertopic, question_short)
    with outlier_tab:
        stats = get_doc_stats(doc_infos)
        display_outliers_topic(doc_infos, custom_bertopic, stats)


def write():
    st.write("## Evaluation des topics générés")
    options = ["transition_ecologique", "solutions_violence_enfants", "MDPH_MDU_negatif", "MDPH_MDU_positif", "mesure_transition_ecologique"]
    question_short = st.selectbox("Choisissez la question à analyser :", options=options)
    #st.write("### Question : Quelle est pour vous la mesure la plus importante pour réussir la transition écologique ? C’est la dernière question, partagez-nous toutes vos idées !")
    model_path = "data/topic_modeling/" + question_short + "/bertopic_model"
    
    # Data Prep
    custom_bertopic = load_model(model_path)
    filepath = "data/topic_modeling/" + question_short + "/doc_infos.csv"
    doc_infos = prep_doc_info(load_doc_infos(filepath))
    cleaned_labels = load_cleaned_labels(question_short, TOPIC_FOLDER)
    stats = get_doc_stats(doc_infos)
    
    display_topic_overview(custom_bertopic, stats)
    
    
    topic_selection(custom_bertopic, doc_infos, cleaned_labels, question_short)
    return


if __name__ == "__main__":
    st.set_page_config(
        layout="wide", page_icon="📊", page_title="Agora -- NLP"
    )
    write()
