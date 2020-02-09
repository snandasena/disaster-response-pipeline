import sys

sys.path.append("common")
from train_classifier import *

staring_verb_extractor = StartingVerbExtractor()
verb_count_extractor = VerbCountExtractor()
starting_modal_extractor = StartingModalExtractor()
noun_count_extractor = NounCountExtractor()


def test_load_data(file_name):
    return load_data(file_name)


def test_stating_verb_extract(text):
    print(staring_verb_extractor.starting_verb(text))


def test_transform(X):
    print(staring_verb_extractor.transform(X))


def test_tokenize(text):
    print(tokenize_text(text))


def test_total_verb_counts(text):
    print(verb_count_extractor.count_verbs(text))


def test_stating_modals(text):
    print(starting_modal_extractor.starting_modals(text))


def test_total_noun_counts(text):
    print(noun_count_extractor.count_nouns(text))


if __name__ == "__main__":
    debug_data = ['.', './data/DisasterResponse.db', './models/classifier.pkl']

    X, Y, col_names = load_data(debug_data[1])
    # for text in X[:100].values:
    #     test_tokenize(text)

    # for text in texts:
    #     test_stating_verb_extract(text)
    #
    # test_transform(X)
    # for text in X[:100].values:
    #     test_total_verb_counts(text)
    #
    # for text in X[:100].values:
    #     test_total_noun_counts(text)
    #
    for text in X[:100].values:
        test_stating_modals(text)
