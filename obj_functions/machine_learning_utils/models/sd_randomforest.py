from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer


def gini(truth, pred):
    g = np.array(np.c_[truth, pred, np.arange(len(truth))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1*g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)


def gini_sklearn(truth, pred):
    return gini(truth, pred) / gini(truth, truth)


def evaluate_safedriver(hp_dict, train_data, valid_data):
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", **hp_dict)

    train_X = train_data.drop(["id", "target"], axis=1)
    train_target = train_data["target"]
    valid_X = valid_data.drop(["id", "target"], axis=1)
    valid_target = valid_data["target"]
    gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)

    clf.fit(train_X, train_target)
    gini_score = gini_scorer(clf, valid_X, valid_target)

    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", **hp_dict)
    gini_score = cross_val_score(clf, train_X, train_target, scoring=gini_scorer, cv=StratifiedKFold(n_splits=5)).mean()
    print("### {} ###".format(gini_score))

    # minimization
    return {"gini": 1. - gini_score}
