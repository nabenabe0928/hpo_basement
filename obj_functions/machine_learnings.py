import machine_learning_utils as ml_utils
import models
import datasets


def train(model, hp_dict, train_data, test_data, cuda_id, save_path):
    print(hp_dict)
    loss_min, acc_max = ml_utils.start_train(model, train_data, test_data, cuda_id, save_path)
    return {"error": 1. - acc_max, "cross_entropy": loss_min}


def cnn(hp_dict, cuda_id, save_path, experimental_settings):
    model = models.CNN(**hp_dict)
    train_data, test_data = datasets.get_data(dataset_name=experimental_settings["dataset_name"],
                                              batch_size=hp_dict["batch_size"],
                                              n_cls=experimental_settings["n_cls"],
                                              image_size=experimental_settings["image_size"],
                                              sub_prop=experimental_settings["sub_prop"],
                                              biased_cls=experimental_settings["biased_cls"]
                                              )
    return train(model, hp_dict, train_data, test_data, cuda_id, save_path)
