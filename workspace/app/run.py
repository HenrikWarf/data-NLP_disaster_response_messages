import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster-data.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    number_of_data_points = len(df)

    # Graph number 2
    df_y_data = df.iloc[:,4:]
    sum_of_data = df_y_data.sum().sort_values(ascending=False)
    labels = list(sum_of_data.index)
    label_count = list(sum_of_data.values)
    
    # create visuals
    # TODO
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'width': 600,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
             'data': [
                Bar(
                    x=labels,
                    y=label_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message per Label',
                'width': 800,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label"
                }
            }

        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', 
        ids=ids, 
        graphJSON=graphJSON,
        number_of_data_points=number_of_data_points
        )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()