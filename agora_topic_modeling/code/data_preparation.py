import re
import pandas as pd


def fracking(df: pd.DataFrame, col: str, sep: str)-> pd.DataFrame:
    # lambda split that filter empty elements from list
    custom_filter = lambda x: x not in ["", " ", None]
    custom_split = lambda x: list(filter(custom_filter, x.split(sep)))

    df = df.copy()
    df["fracked_text"] = df[col].apply(custom_split)
    print(len(df.index))
    df = df.explode("fracked_text")
    print(len(df.index))
    df["old_index"] = df.index
    fracking_count = df.groupby("old_index").agg(fracking_count=("old_index", "count")).reset_index()
    df = df.merge(fracking_count, on="old_index")
    print(len(df.index))
    return df

def compute_response_size(df: pd.DataFrame, response_col: str)-> pd.DataFrame:
    df_ = df.copy()
    df_["response_size"] = df_[response_col].str.split().apply(len)
    return df_


# Remove, useless answers
def prep_answer_df(df: pd.DataFrame, response_col: str):
    cleaned_df = df.copy()
    NUMBERED_LIST_HANDLER = lambda x: re.sub(r'([0-9]+\. )', ' ', x)
    SPECIAL_CHAR_HANDLER = lambda x: re.sub(r'[()/\\]', ' ', x)
    WHITESPACE_HANDLER = lambda x: re.sub('\s+', ' ', re.sub('\n+', ' ', x.strip()))
    cleaned_df["cleaned_text"] = cleaned_df[response_col].apply(NUMBERED_LIST_HANDLER)
    cleaned_df["cleaned_text"] = cleaned_df["cleaned_text"].apply(SPECIAL_CHAR_HANDLER)
    cleaned_df["cleaned_text"] = cleaned_df["cleaned_text"].apply(WHITESPACE_HANDLER)
    
    fracked_df = fracking(cleaned_df, "cleaned_text", sep=".")
    fracked_df = fracked_df.dropna()
    fracked_df = compute_response_size(fracked_df, "fracked_text")
    return fracked_df


def get_cleaned_doc_from_question(df: pd.DataFrame, question_col: str, response_col: str, question: str)-> pd.DataFrame:
    df_filtered = df[df[question_col] == question].copy()
    cleaned_doc = prep_answer_df(df_filtered, response_col)
    return cleaned_doc
