from wordcloud import WordCloud
from bertopic import BERTopic
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def create_wordcloud_from_topic(word_freq: pd.DataFrame):
    dict = word_freq[["word", "freq"]].to_dict("index")
    text = {dict[row]['word']: dict[row]["freq"] for row in dict}
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    return wc


def display_topic_histogram(trained_bert: BERTopic):
    trained_bert.visualize_barchart()
