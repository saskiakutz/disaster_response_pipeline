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