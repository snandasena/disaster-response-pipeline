#!/bin/bash
# to install required python dependencies
python -m pip install -r requirements.txt

# to install nltk packages
python -m nltk.downloader wordnet
python -m nltk.downloader stopwords
python -m nltk.downloader punkt

# to run ETL pipeline
rm -rf data/DisasterResponse.db >/dev/null

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

# to run ML pipeline

rm -rf models/classifier.pkl >/dev/null
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
