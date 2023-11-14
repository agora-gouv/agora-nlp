import string

import spacy
import unicodedata
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
from spacymoji import Emoji
nltk.download("stopwords")
nltk.download("wordnet")


def strip_accents(text: str):
    cleaned_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return cleaned_text


def get_tokens_from_doc(doc: str, with_stop=True):
    # TODO: Try pattern matching / mapreduce when test are done
    list_token = []
    for word in doc:
        # if not word.is_punct and not word.is_space and not word.is_stop and not word.like_url and word.lemma_!="":
        if (not word.is_punct or word._.is_emoji) and not word.is_space and not word.like_url and word.lemma_!="":
            if with_stop or (not with_stop and not word.is_stop):
                if word._.is_emoji:
                    list_token.append(word.lemma_)
                else:
                    list_token.append(strip_accents(word.lemma_.lower()))
    return list_token
    

def clean_df_col(df: pd.DataFrame, question_col: str, token_col: str):
    nlp = spacy.load("fr_core_news_sm")
    nlp.add_pipe("emoji", first=True)
    df[question_col].fillna("",inplace=True)
    df[token_col] = df[question_col].apply(
        lambda x: get_tokens_from_doc(nlp(str(x)), False))        
    return df


def text_process_spacy(text: str, nlp)-> list[str]:
    return get_tokens_from_doc(nlp(text), False)


def text_process_nltk(text: str, strip_accent: bool=True)-> list[str]:
    stemmer = WordNetLemmatizer()
    # Remove punctuation
    cleaned_text = [char for char in text if char not in string.punctuation]
    # Remove digits
    cleaned_text = "".join([i for i in cleaned_text if not i.isdigit()])
    # Remove stopwords and Put all letters in lowercase
    cleaned_text = [word.lower() for word in cleaned_text.split() if word not in stopwords.words("french")]
    # Keep only lemmatized form of word 
    cleaned_text = [stemmer.lemmatize(word) for word in cleaned_text]
    if strip_accent:
        cleaned_text = [unidecode(word) for word in cleaned_text]
    return cleaned_text
