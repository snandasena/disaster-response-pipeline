import json
import sys
from itertools import product

import plotly
from flask import Flask
from flask import render_template, request
from plotly.graph_objects import Heatmap, Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

sys.path.append('../common')
from common.nlp_common_utils import *

app = Flask(__name__)


def tokenize(text):
    return tokenize_text(text)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
df_categories = df.iloc[:, 4:].sum()
# load model
model = joblib.load("../models/classifier.pkl")


def generate_graph_with_template(data, title, yaxis_title, xaxi_title):
    return {
        'data': [data],

        'layout': {
            'title': title,
            'yaxis': {
                'title': yaxis_title
            },
            'xaxis': {
                'title': xaxi_title
            }
        }
    }


def generate_message_genres_bar_chart():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    data = Bar(x=genre_names, y=genre_counts)
    title = 'Distribution of Message Genres'
    y_title = 'Count'
    x_title = 'Genre'
    return generate_graph_with_template(data, title, y_title, x_title)


def generate_message_categories_distribution_bar_chart():
    data = Bar(x=[s.replace("_", " ") for s in df_categories.index],
               y=list(df_categories.sort_values(ascending=False)))
    title = 'Distribution of Message Categories'
    y_title = 'Count'
    x_title = 'Category'

    return generate_graph_with_template(data, title, y_title, x_title)


def generate_two_cat_relation_heat_map():
    data = Heatmap(
        z=df.iloc[:, 4:].corr(),
        y=df.iloc[:, 4:].columns,
        x=df.iloc[:, 4:].columns)

    title = 'Correlation Distribution of Categories'
    y_title = 'Category'
    x_title = 'Category'
    return generate_graph_with_template(data, title, y_title, x_title)


def generate_graphs():
    # create visuals
    graphs = [generate_message_genres_bar_chart(),
              generate_message_categories_distribution_bar_chart(),
              generate_two_cat_relation_heat_map()]

    return graphs


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = generate_graphs()
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


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
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
