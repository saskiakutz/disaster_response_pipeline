# A Disaster Response Pipeline

This disaster response pipeline classifies disaster messages collected by Figure Eight.

The pipeline consists of three parts:

1. ETL pipeline: a data cleaning pipeline
2. ML pipeline: a machine learning pipeline
3. Flask Web App: a data visualisation web app

## ETL pipeline

The ETL pipeline loads and cleans the original datasets provided by Figure Eight. The final output is a single dataframe
stored in an SQL database. All necessary files are stored in the data folder.

- categories.csv
- messages.csv
- DisasterResponse.db
- process_data.py

In order to run the ETL pipeline enter the following command into your
terminal: `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`.

## ML pipeline

THe ML pipeline loads the data stored at the end of the ETL pipeline. It then trains and tests the ML model. In the
final step the ML pipeline stores the ML model as a pickle file.

The ML model consists of the following parts: 
- CountVectorizer with a tokenizer that prepares the messages for adequate NLP
- TfidfTransformer
- SGDClassifier, a Linear Support Vector Machine

In order to run the ML pipeline enter the following command into your
terminal: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`.