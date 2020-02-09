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

sys.path.append('common')
from common.nlp_common_utils import *


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from DisasterResponse", engine)
    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns.values


def tokenize(text):
    return tokenize_text(text)


def build_pipeline(estimator):
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
    if params is None:
        params = parameters
    cv = GridSearchCV(pipeline, param_grid=params)

    return cv


def build_model(enable_param_tuning=False):
    estimator = MultiOutputClassifier(OneVsRestClassifier(RandomForestClassifier(n_jobs=5)))
    pipeline = build_pipeline(estimator)
    if enable_param_tuning:
        return hyper_param_tuning(pipeline)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    prediction = model.predict(X_test)
    print(classification_report(Y_test, prediction, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(enable_param_tuning=True)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        # evaluate_model(model, X_test, Y_test, category_names)

        # print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    debug_data = ['.', './data/DisasterResponse.db', './models/classifier.pkl']
    sys.argv = debug_data
    main()
