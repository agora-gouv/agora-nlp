import os

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
DAGS_FOLDER = os.path.join(PROJECT_FOLDER, 'dags')
PREDICTIONS_FOLDER = os.path.join(DATA_FOLDER, 'predictions')

FILEPATH = os.path.join(DATA_FOLDER, 'data.parquet')
CLEANED_DATA_FILEPATH = os.path.join(DATA_FOLDER, 'cleaned_data.parquet')
DOC_INFOS_FILEPATH = os.path.join(DATA_FOLDER, 'doc_infos.parquet')
DOC_INFOS_WITH_SUB_FILEPATH = os.path.join(DATA_FOLDER, 'doc_infos_with_sub.parquet')

DATABASE_CONFIG_FILEPATH = os.path.join(DAGS_FOLDER, 'database.ini')

# percentage of answer from a topic needs to be higher than this to get subtopics
TOPIC_THRESHOLD_FOR_SUBTOPIC = 5