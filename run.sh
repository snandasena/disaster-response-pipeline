#!/bin/bash
# to install required python dependencies
python -m pip install -r requirements.txt

# to install nltk packages
python -m nltk.downloader wordnet
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

# to run ETL pipeline
rm -rf ./data/DisasterResponse.db >/dev/null
#
python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db

# to run ML pipeline

rm -rf ./models/classifier.pkl >/dev/null
python train_classifier.py ./data/DisasterResponse.db ./models/classifier.pkl

# run the application
# shellcheck disable=SC2164
python run.py ./data/DisasterResponse.db ./models/classifier.pkl
