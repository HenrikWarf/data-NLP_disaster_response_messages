import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine  
from nltk.tokenize import word_tokenize
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import nltk
nltk.download('stopwords')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = list(y.columns)

    return X, Y, category_names


def tokenize(text):
        
    #remove punctuation and make lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize the text
    words = word_tokenize(text)
    
    #remove stopwords
    clean_words = []
    for w in words:
        if w not in stopwords.words("english"):
            clean_words.append(w)
    
    lemmatizer = WordNetLemmatizer()
    
    words_final = []
    for w in clean_words:
        clean_tok = lemmatizer.lemmatize(w).strip()
        words_final.append(clean_tok)
    
    return words_final


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_test_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_test_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()