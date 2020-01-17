import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from obj_functions.machine_learning_utils.datasets import dataset_utils


# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


def load_dataset(data_frac):
    raw_data = pd.read_csv('toxic-comment/train.csv').fillna(' ')
    n_raw = len(raw_data)

    n_train = int(n_raw * data_frac)
    print('Loaded')

    return raw_data.head(n_train)


def get_word_vec(train_text):
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        max_features=50000)
    word_vectorizer.fit(train_text)
    print('Word TFIDF')
    train_word_features = word_vectorizer.transform(train_text)

    return train_word_features


def get_char_vec(train_text):
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        max_features=50000)
    char_vectorizer.fit(train_text)
    print('Char TFIDF')
    train_char_features = char_vectorizer.transform(train_text)

    return train_char_features


def get_toxic(experimental_settings):
    data_frac = dataset_utils.dataset_check_for_kaggle(experimental_settings, "toxic-comment", "toxic_lgbm")

    train = load_dataset(data_frac)
    train_text = train['comment_text']
    train_word_features = get_word_vec(train_text)
    train_char_features = get_char_vec(train_text)

    train_features = hstack([train_char_features, train_word_features])
    train_features.tocsr()
    print('HStack')

    train.drop('comment_text', axis=1, inplace=True)

    return [train_features, train]
