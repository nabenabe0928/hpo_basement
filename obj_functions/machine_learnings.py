import obj_functions.machine_learning_utils as ml_utils
from obj_functions import models, datasets


def evaluating_model(model, hp_dict, train_data, test_data, cuda_id, save_path):
    print(hp_dict)
    loss_min, acc_max = ml_utils.start_train(model, train_data, test_data, cuda_id, save_path)
    return {"error": 1. - acc_max, "cross_entropy": loss_min}


def cnn(experimental_settings):
    train_dataset, test_dataset = datasets.get_dataset(experimental_settings)

    def _imp(hp_dict, cuda_id, save_path):
        model = models.CNN(**hp_dict, n_cls=experimental_settings.n_cls)
        train_data, test_data = datasets.get_data(train_dataset, test_dataset, batch_size=model.batch_size)
        return evaluating_model(model, hp_dict, train_data, test_data, cuda_id, save_path)

    return _imp


def wrn(experimental_settings):
    train_dataset, test_dataset = datasets.get_dataset(experimental_settings)

    def _imp(hp_dict, cuda_id, save_path):
        model = models.WideResNet(**hp_dict, n_cls=experimental_settings.n_cls)
        train_data, test_data = datasets.get_data(train_dataset, test_dataset, batch_size=model.batch_size)
        return evaluating_model(model, hp_dict, train_data, test_data, cuda_id, save_path)

    return _imp


def dnbc(experimental_settings):
    train_dataset, test_dataset = datasets.get_dataset(experimental_settings)

    def _imp(hp_dict, cuda_id, save_path):
        model = models.DenseNetBC(**hp_dict, n_cls=experimental_settings.n_cls)
        train_data, test_data = datasets.get_data(train_dataset, test_dataset, batch_size=model.batch_size)
        return evaluating_model(model, hp_dict, train_data, test_data, cuda_id, save_path)

    return _imp
