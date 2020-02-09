# import libraries
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# starting verbs extractor
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    @staticmethod
    def starting_verb(text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize_text(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP']:
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class VerbCountExtractor(BaseEstimator, TransformerMixin):

    @staticmethod
    def count_verbs(text):
        total_verbs = 0
        pos_tags = nltk.pos_tag(tokenize_text(text))
        for word, tag in pos_tags:
            if tag in ['VB', 'VBP', 'VBD']:
                total_verbs = total_verbs + 1

        return total_verbs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.count_verbs)
        return pd.DataFrame(X_tagged)


class NounCountExtractor(BaseEstimator, TransformerMixin):
    @staticmethod
    def count_nouns(text):
        total_nouns = 0
        pos_tags = nltk.pos_tag(tokenize_text(text))
        for word, tag in pos_tags:
            if tag in ['NN', 'NNS', 'NNPS', 'NNP']:
                total_nouns = total_nouns + 1

        return total_nouns

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.count_nouns)
        return pd.DataFrame(X_tagged)


class StartingModalExtractor(BaseEstimator, TransformerMixin):
    @staticmethod
    def starting_modals(text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize_text(sentence))
            if len(pos_tags) != 0:
                first_word, first_tag = pos_tags[0]
                if first_tag == 'MD':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_modals)
        return pd.DataFrame(X_tagged)


def tokenize_text(text):
    # transform to lower case
    text = text.lower()
    # remove urls
    text = re.sub(url_regex, " ", text)
    # remove punctuations
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    # remove numbers
    text = re.sub(r"\d+", " ", text)
    # tokenize for words
    words = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words("english")]
    # lemmatize the tokens
    clean_tokens = list(map(lambda token: lemmatizer.lemmatize(token).strip(), tokens))
    return clean_tokens
