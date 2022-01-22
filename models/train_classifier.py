import sys
import nltk

nltk.download(['punkt', 'stopwords', 'wordnet'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier


def load_data(database_filepath):
    """
    Loading the data file from the sqlite database
    :param database_filepath: name of the sqlite database
    :return: list with the message column, the labels and the names of the labels
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data_frame', engine)

    X = df['message'].values
    Y = df[df.columns[4:]]
    Y_names = df.columns[4:]
    return [X, Y, Y_names]


def tokenize(text):
    """
    Tokenization of text
    :param text: here, the messages
    :return: tokenzied text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    urls = re.findall(url_regex, text)
    for url in urls:
        text = text.replace(url, 'urlplaceholder')
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = word_tokenize(text)

    token_wo_stop = []

    for token in tokens:
        if token not in stopwords.words('english'):
            token_wo_stop.append(token)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok.lower().replace(' ', ''))
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Setup of the model. The pipeline consists of a CountVectorizer, a TfidfTransformer and a MultiOutputClassifier with
    a SGDClassifier. The model is optimized with a GridSearch.
    :return: optimize model
    """
    pipeline_LSV = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('multi', MultiOutputClassifier(SGDClassifier()))])

    parameters = {'multi__estimator__loss': ['hinge', 'perceptron'],
                  'tfidf__smooth_idf': [True, False],
                  'tfidf__sublinear_tf': [True, False]}

    model = GridSearchCV(pipeline_LSV, parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluation of the model against the test data.
    :param model: build model
    :param X_test: dataset of test messages
    :param Y_test: dataset of test labels/categories
    :param category_names: names of the labels/categories of the text data
    :return: classification report
    """
    y_pred = model.predict(X_test)
    classification_report_multi(Y_test, y_pred, category_names)


def classification_report_multi(ytest, ypred, targetnames):
    """
    Printing the classification report
    :param ytest: test labels/categories data
    :param ypred: predicted labels based on the test messages
    :param targetnames: the names of the labels/categories
    :return: print of the report
    """
    for i in range(0, len(targetnames)):
        print('Category {}'.format(targetnames[i]))
        report = classification_report(ytest[targetnames[i]], ypred[:, i], zero_division=1)
        print(report)


def save_model(model, model_filepath):
    """
    Saving of the model as a pkl file
    :param model: build and trained model
    :param model_filepath: path for file
    :return:
    """
    dbfile = open(model_filepath, 'ab')
    pickle.dump(model, dbfile)
    dbfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
