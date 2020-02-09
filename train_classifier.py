import pickle
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine

# append common util to `this` runtime
sys.path.append('common')
from common.nlp_common_utils import *


def load_data(database_filepath):
    """
        Load data from SQLite database and split into features and target.

    INPUT:
        database_filepath - SQLite database location

    OUTPUT:
        X -- features in the form of a message
        Y -- multiple target variables
        category_names -- list of all target variable names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from DisasterResponse", engine)
    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns.values


def tokenize(text):
    """
    Used a common utility functions for tokenize text in to cleaned token list.

    INPUT:
        text - raw message

    OUTPUT:
        clean_tokens -- cleaned tokenized list
    """
    return tokenize_text(text)


def build_pipeline(estimator):
    """
    This separate function can be used to test with multiple estimators.
    INPUT:
        estimator - An estimator that is used to build `sklearn.pipeline` Pipeline

    OUTPUT:
        pipeline - `sklearn.pipeline` with a estimator and multiple features.
    """

    # define pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb_noun', StartingVerbExtractor()),
            ('count_verbs', VerbCountExtractor()),
            ('count_nouns', NounCountExtractor()),
            ('starting_modals', StartingModalExtractor())
        ])),

        ('clf', estimator)
    ])

    return pipeline


# general RandomForestClassifier hyper params and NLP pipeline params
parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    'features__text_pipeline__vect__max_features': (None, 10, 100),
    'features__text_pipeline__tfidf__use_idf': (True, False),
    'clf__estimator__estimator__n_estimators': [10, 20, 100],
    'clf__estimator__estimator__min_samples_split': [2, 3, 4],
    'features__transformer_weights': (
        {'text_pipeline': 1, 'starting_verb': 0.5},
        {'text_pipeline': 0.5, 'starting_verb': 1},
        {'text_pipeline': 0.8, 'starting_verb': 1},
    )
}


def hyper_param_tuning(pipeline, params=None):
    """
    This function can be used to tune hyper params for an estimator.

    INPUT:
        pipeline - `sklearn.pipeline` Pipeline, which is already created.
        params - hyper parameters based on that is used to create a pipeline.

    OUTPUT:
        cv - `GridSearchCV`
    """
    if params is None:
        params = parameters
    cv = GridSearchCV(pipeline, param_grid=params)

    return cv


def build_model(enable_param_tuning=False):
    """
    Model that handles multi-output classification and use grid search.

    INPUT:
        enable_param_tuning - this required to enable for parameter tuning.

    OUTPUT:
        pipeline model - model that created with params or non params with a selected estimator.
    """

    # multioutput estimator with random forest classifier
    estimator = MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier(n_jobs=5)))
    # build pipeline
    pipeline = build_pipeline(estimator)
    if enable_param_tuning:
        return hyper_param_tuning(pipeline)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict values and key metrics are presented for each category.

    INPUT:
        model - classification model used for prediction
        X_test - features in the form of a message for test set
        y_test - multiple target variables for test set
        category_names - list of all target variable names

    OUTPUT:
        classification_report's console output

    """
    prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    INPUT:
        model - classification model to be saved
        model_filepath - file path to save model as
    OUTPUT:
        a pickle binary file
    """
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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    if len(sys.argv) == 0:
        sys.argv = ['.', './data/DisasterResponse.db', './models/classifier.pkl']
    main()
