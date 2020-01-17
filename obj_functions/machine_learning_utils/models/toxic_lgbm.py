import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


# reference: https://www.kaggle.com/peterhurford/lightgbm-with-select-k-best-on-tfidf


def evaluate_toxic(hp_dict, train_data):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_features = train_data[0]
    train_labels = train_data[1]
    scores = {}

    for class_name in class_names:
        print(class_name)
        train_target = train_labels[class_name]
        model = LogisticRegression(solver='sag')
        sfm = SelectFromModel(model, threshold=0.2)

        print(train_features.shape)
        train_sparse_matrix = sfm.fit_transform(train_features, train_target)
        print(train_sparse_matrix.shape)
        d_train = lgb.Dataset(train_sparse_matrix, label=train_target)

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
        score = lgb.cv(params,
                       train_set=d_train,
                       nfold=5,
                       stratified=True,
                       num_boost_round=rounds_lookup[class_name],
                       verbose_eval=10)
        cv_auc = max(score["auc-mean"])
        scores[class_name] = 1. - cv_auc
    scores["mean"] = np.array(list(scores.values())).mean()

    return scores
