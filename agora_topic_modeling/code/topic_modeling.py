import pandas as pd
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


from nltk.corpus import stopwords
from random import sample

# Create instances of GPU-accelerated UMAP and HDBSCAN
# from cuml.cluster import HDBSCAN
# from cuml.manifold import UMAP

# umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
# hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)


def get_lightweight_bertopic_model(X: pd.Series)-> tuple[BERTopic, pd.DataFrame]:
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer_model = CountVectorizer(stop_words=stopwords.words("french"), strip_accents="ascii")
    pipe = make_pipeline(
        TfidfVectorizer(),
        TruncatedSVD(100)
    )
    nr_topics = 10
    min_topic_size = 100
    topic_model = BERTopic(embedding_model=pipe, vectorizer_model=vectorizer_model, nr_topics=nr_topics, min_topic_size=min_topic_size, language="french", verbose=True)
    print("Fit Transform")
    topics, probs = topic_model.fit_transform(X)
    print("Returns")
    return topic_model, topics


def get_custom_bertopic_model(X: pd.Series)-> tuple[BERTopic, pd.DataFrame]:
    # Remove stopwords
    print("Vectorized model")
    vectorizer_model = CountVectorizer(stop_words=stopwords.words("french"), strip_accents="ascii")
    embedding_model = SentenceTransformer("/Users/theo.santos/Documents/Missions/Agora/models")

    cluster_model = KMeans(n_clusters=5)
    
    nr_topics = "auto"
    #nr_topics = 10
    min_topic_size = 10
    if True:
        print("Topic model")
        topic_model = BERTopic(hdbscan_model=cluster_model, embedding_model=embedding_model, vectorizer_model=vectorizer_model, nr_topics=nr_topics, min_topic_size=min_topic_size, language="french", verbose=True)
    else:
        n_docs = round(X.size * 0.01)
        topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=n_docs, language="french")
    
    print("Fit Transform")
    topics, probs = topic_model.fit_transform(X)
    print("Returns")
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
    # if topic_range == 0:
    #     #doc_infos["sub_topic"] = -2
    #     return doc_infos
    to_merge = None
    for topic in range(topic_range):
        print(f"Sub_topics for topic {topic}")
        topic_documents = get_docs_from_topic(doc_infos, topic)
        X = topic_documents["Document"]
        custom_bert, custom_topics = get_custom_bertopic_model(X)
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


def get_topics_and_subtopics_from_df(df: pd.DataFrame, col_to_analyse: str):
    fracking_merge = df[["old_index", "fracking_count"]].copy()
    X = df[col_to_analyse]
    custom_bert, custom_topics = get_custom_bertopic_model(X)
    doc_infos = custom_bert.get_document_info(X)
    doc_infos = doc_infos.join(fracking_merge)
    answers_per_topic = get_topic_distribution(doc_infos)
    #sub_topic_range = answers_per_topic[answers_per_topic["percentage"] > float(os.getenv("TOPIC_THRESHOLD_FOR_SUBTOPIC"))].count()[0]

    doc_infos_w_subtopics = get_sub_topics(doc_infos, 2)
    return doc_infos_w_subtopics
