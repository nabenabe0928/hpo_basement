import pandas as pd
from obj_functions.machine_learning_utils.datasets import dataset_utils
from sklearn.model_selection import train_test_split


# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data


def get_safedriver(experimental_settings):
    data_frac = dataset_utils.dataset_check_for_kaggle(experimental_settings, "safe-driver", "sd_randomforest")
    raw_data = pd.read_csv('safe-driver/train.csv')

    if data_frac < 1.:
        train, _, _, _ = train_test_split(raw_data, raw_data["target"], test_size=1. - data_frac, stratify=raw_data["target"])
    else:
        train = raw_data
    print('Loaded')

    return train
