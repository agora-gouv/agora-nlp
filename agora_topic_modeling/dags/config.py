import os

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
PREDICTIONS_FOLDER = os.path.join(DATA_FOLDER, 'predictions')

FILEPATH = os.path.join(DATA_FOLDER, 'data.csv')
CLEANED_DATA_FILEPATH = os.path.join(DATA_FOLDER, 'cleaned_data.csv')