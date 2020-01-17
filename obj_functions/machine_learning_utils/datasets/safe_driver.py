import pandas as pd
from obj_functions.machine_learning_utils.datasets import dataset_utils
from sklearn.model_selection import train_test_split


# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data


def get_safedriver(experimental_settings):
    data_frac = dataset_utils.dataset_check_for_kaggle(experimental_settings, "safe-driver", "sd_randomforest")

    raw_data = pd.read_csv('safe-driver/train.csv')
    # n_raw = len(raw_data)

    train, valid, _, _ = train_test_split(raw_data, raw_data["target"], test_size=0.2, stratify=raw_data["target"])
    # train = raw_data.head(int(0.8 * n_raw))
    # valid = raw_data.tail(n_raw - int(0.8 * n_raw))

    n_train = int(len(train) * data_frac)
    print('Loaded')

    return train.head(n_train), valid
