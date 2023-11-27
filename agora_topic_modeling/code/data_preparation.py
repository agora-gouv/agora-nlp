import re
import pandas as pd

def compute_response_size(df: pd.DataFrame, response_col: str)-> pd.DataFrame:
    df_ = df.copy()
    df_["response_size"] = df_[response_col].str.split().apply(len)
    return df_    


# Remove, useless answers
def prep_answer_df(df: pd.DataFrame, response_col: str):
    cleaned_df = df.copy()
    cleaned_df = compute_response_size(cleaned_df, response_col)
    WHITESPACE_HANDLER = lambda k: re.sub('\\s+', ' ', re.sub('\n+', ' ', k.strip()))
    cleaned_df["cleaned_text"] = cleaned_df["Response Text"].apply(WHITESPACE_HANDLER)
    return cleaned_df


def get_cleaned_doc_from_question(df: pd.DataFrame, question_col: str, response_col: str, question: str)-> pd.DataFrame:
    df_filtered = df[df[question_col] == question].copy()
    cleaned_doc = prep_answer_df(df_filtered, response_col)
    return cleaned_doc
