import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# reference: https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf


def evaluate_toxic(hp_dict, train_data, valid_data):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_features = train_data[0]
    valid_features = valid_data[0]
    train_labels = train_data[1]
    valid_labels = valid_data[1]
    scores = {}

    for class_name in class_names:
        print(class_name)
        train_target = train_labels[class_name]
        model = LogisticRegression(solver='sag')
        sfm = SelectFromModel(model, threshold=0.2)
        print(train_features.shape)
        train_sparse_matrix = sfm.fit_transform(train_features, train_target)
        valid_sparse_matrix = sfm.transform(valid_features)
        valid_target = valid_labels[class_name]
        print(train_sparse_matrix.shape)
        # train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target, test_size=0.05)
        d_train = lgb.Dataset(train_sparse_matrix, label=train_target)
        d_valid = lgb.Dataset(valid_sparse_matrix, label=valid_target)
        watchlist = [d_train, d_valid]
        params = {**hp_dict,
                  'application': 'binary',
                  'verbosity': -1,
                  'metric': 'auc',
                  'data_random_seed': 2,
                  'nthread': 4}
        rounds_lookup = {'toxic': 140,
                         'severe_toxic': 50,
                         'obscene': 80,
                         'threat': 80,
                         'insult': 70,
                         'identity_hate': 80}
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=rounds_lookup[class_name],
                          valid_sets=watchlist,
                          verbose_eval=10)
        score = 1. - roc_auc_score(valid_labels[class_name], model.predict(valid_sparse_matrix))
        scores[class_name] = score
    scores["mean"] = np.array(list(scores.values())).mean()

    return scores
