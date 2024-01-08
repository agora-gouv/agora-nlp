import pandas as pd

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


from transformers import GenerationConfig
from nltk.corpus import stopwords
from random import sample



def get_custom_bertopic_model(X: pd.Series)-> tuple[BERTopic, pd.DataFrame]:
    # Remove stopwords
    vectorizer_model = CountVectorizer(stop_words=stopwords.words("french"), strip_accents="ascii")
    
    #nr_topics = "auto"
    nr_topics = 10
    min_topic_size = 200
    if True:
        topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=nr_topics, min_topic_size=min_topic_size, language="french")
    else:
        n_docs = round(X.size * 0.01)
        topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=n_docs, language="french")
    
     
    topics, probs = topic_model.fit_transform(X)
    return topic_model, topics



def get_generation_config(model):
    generation_config = GenerationConfig(
        max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
    )
    return generation_config


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


# Sous_topics
def get_sub_topics(doc_infos: pd.DataFrame, topic_range: int):
    to_merge = None
    for topic in range(topic_range):
        print(f"Sub_topics for topic {topic}")
        topic_documents = get_docs_from_topic(doc_infos, topic)
        X = topic_documents["Document"]
        custom_bert, custom_topics = get_custom_bertopic_model(X, True)
        topic_infos = custom_bert.get_document_info(X)
        topic_infos["id"] = topic_documents.index
        print(get_topic_distribution(topic_infos))
        topic_infos = topic_infos.rename(columns={"Topic": "sub_topic", "Name": "sub_name"})
        if to_merge is None:
            to_merge = topic_infos[["id", "sub_topic", "sub_name"]]
        else:
            to_merge = pd.concat([to_merge, topic_infos[["id", "sub_topic", "sub_name"]]])
    doc_infos["id"] = doc_infos.index
    doc_infos_w_subtopics = doc_infos.merge(to_merge, on="id", how="left")
    doc_infos_w_subtopics["sub_topic"] = doc_infos_w_subtopics["sub_topic"].fillna(-2)
    return doc_infos_w_subtopics
