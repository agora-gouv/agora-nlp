import pandas as pd
import numpy as np
from transformers import pipeline


def load_doc_infos(filepath: str) -> pd.DataFrame:
    doc_infos = pd.read_csv(filepath, index_col=0)
    doc_infos["Topic"] = doc_infos["Topic"].astype(int)
    doc_infos = doc_infos.sort_values("Probability", ascending=False)
    return doc_infos


#sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def get_sentiment_pipeline():
    sentiment_pipeline = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", max_length=512, truncation=True)
    return sentiment_pipeline


def add_sentiment_analysis_results_to_df(doc_info: pd.DataFrame, sentiments: list[dict]):
    doc_info_w_sentiment = doc_info.copy()
    doc_info_w_sentiment["sentiment"] = list(map(lambda x: x.get("label"), sentiments))
    doc_info_w_sentiment["sentiment_score"] = list(map(lambda x: x.get("score"), sentiments))
    # doc_info_w_sentiment["is_high_score"] = doc_info_w_sentiment["score"] > 0.75
    return doc_info_w_sentiment


def apply_sentiment_analysis(doc_infos: pd.DataFrame):
    sentiment_pipeline = get_sentiment_pipeline()
    X = doc_infos["Document"].values
    result = sentiment_pipeline(list(X))
    doc_info_w_sentiment = add_sentiment_analysis_results_to_df(doc_infos, result)
    return doc_info_w_sentiment