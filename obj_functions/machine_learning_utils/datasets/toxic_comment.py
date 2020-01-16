import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from obj_functions.machine_learning_utils.datasets import dataset_utils


# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


def load_dataset(data_frac):
    raw_data = pd.read_csv('toxic-comment/train.csv').fillna(' ')
    n_raw = len(raw_data)

    train = raw_data.head(int(0.8 * n_raw))
    valid = raw_data.tail(n_raw - int(0.8 * n_raw))

    n_train = int(len(train) * data_frac)
    print('Loaded')

    return train.head(n_train), valid


def get_word_vec(train_text, test_text, all_text):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        max_features=50000)
    word_vectorizer.fit(all_text)
    print('Word TFIDF 1/3')
    train_word_features = word_vectorizer.transform(train_text)
    print('Word TFIDF 2/3')
    test_word_features = word_vectorizer.transform(test_text)
    print('Word TFIDF 3/3')

    return train_word_features, test_word_features


def get_char_vec(train_text, test_text, all_text):
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    char_vectorizer.fit(all_text)
    print('Char TFIDF 1/3')
    train_char_features = char_vectorizer.transform(train_text)
    print('Char TFIDF 2/3')
    test_char_features = char_vectorizer.transform(test_text)
    print('Char TFIDF 3/3')

    return train_char_features, test_char_features


def get_toxic(experimental_settings):
    data_frac = dataset_utils.dataset_check_for_kaggle(experimental_settings, "toxic-comment", "toxic_lgbm")

    train, valid = load_dataset(data_frac)
    train_text, valid_text = train['comment_text'], valid['comment_text']
    all_text = pd.concat([train_text, valid_text])
    train_word_features, valid_word_features = get_word_vec(train_text, valid_text, all_text)
    train_char_features, valid_char_features = get_char_vec(train_text, valid_text, all_text)

    train_features = hstack([train_char_features, train_word_features])
    print('HStack 1/2')
    valid_features = hstack([valid_char_features, valid_word_features])
    print('HStack 2/2')

    train.drop('comment_text', axis=1, inplace=True)
    valid.drop('comment_text', axis=1, inplace=True)

    return [train_features, train], [valid_features, valid]
