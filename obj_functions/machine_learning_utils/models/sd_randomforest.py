from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np


def gini(truth, pred):
    g = np.array(np.c_[truth, pred, np.arange(len(truth))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1*g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)


def gini_sklearn(truth, pred):
    return gini(truth, pred) / gini(truth, truth)


def evaluate_safedriver(hp_dict, train_data, valid_data):
    gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", **hp_dict)

    X = train_data.drop(["id", "target"], axis=1)
    y = train_data["target"]
    score = cross_val_score(clf, X, y, scoring=gini_scorer, cv=StratifiedKFold(n_splits=5)).mean()

    return {"gini": 1. - score}