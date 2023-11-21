import pandas as pd
import torch

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from transformers import TFT5ForConditionalGeneration, T5Tokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from random import sample


def get_tokenizer(t5=True):
    if t5:
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    else: 
        tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    return tokenizer


def get_summarizer_pipeline(tokenizer, t5=True):
    if t5:
        language_model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
        summarizer = pipeline("summarization", model=language_model, tokenizer=tokenizer, framework="tf")
    else:
        summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum", tokenizer=tokenizer)
    return summarizer


def get_headline_generator(t5=True):
    if t5: 
        headline_generator = TFT5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
    else:
        headline_generator = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
    return headline_generator


def fit_bertopic_model(docs: list[str], min_topic_size=10):
    topic_model = BERTopic(min_topic_size=min_topic_size, language="french")
    topics = topic_model.fit_transform(docs)
    return topic_model, topics


def get_topic_distribution(doc_infos: pd.DataFrame):
    answers_per_topic = doc_infos.groupby("Topic").agg(answers=("Document", "count")).reset_index()
    answers_per_topic["percentage"] = answers_per_topic["answers"] / answers_per_topic["answers"].values.sum() * 100
    return answers_per_topic


def get_docs_from_topic(doc_infos, topic):
    topic_doc_info = doc_infos[doc_infos["Topic"] == topic].copy()
    return topic_doc_info


def generate_topic_label(answers: list[str], summarizer, tokenizer, headline_generator, verbose=False) -> str:
    summary_list = []
    current_token_length = 0
    max_token_length = 512
    for answer in sample(answers, k=len(answers)):
        max_length = min(150, int(len(answer) / 3))
        summary = summarizer(answer, min_length=10, max_length=max_length)[0]["summary_text"]
        if verbose:
            print("anwser: " + answer)
            print("summary: " + summary)
        current_token_length += len(tokenizer.encode(summary))
        if current_token_length >= max_token_length:
            break
        summary_list.append(summary)

    encoding = tokenizer.encode("titre : " + " ".join(summary_list), return_tensors="pt")
    output = headline_generator.generate(encoding, max_length=64)
    return tokenizer.decode(output[0][1:-1])


def get_headline_from_topics(doc_infos, i_range, verbose=False)-> list[str]:
    tokenizer = get_tokenizer()
    summarizer = get_summarizer_pipeline(tokenizer)
    headline_generator = get_headline_generator()
    topic_labels = []
    for i in range(i_range):
        topic_i = get_docs_from_topic(doc_infos, i)
        doc_i = topic_i['Document'].values.tolist()
        label = "Topic " + str(i) + " : " + generate_topic_label(doc_i, summarizer, tokenizer, headline_generator, verbose)
        print(label)
        topic_labels.append(label)
    return topic_labels
    