import ml_utils
import models


def cnn_train(hp_dict, cuda_id, save_path):
    model = models.CNN(hp_dict)
    print(hp_dict)
    train_data, test_data = #### 
    loss_min, acc_max = ml_utils.start_train(model, train_data, test_data, cuda_id, save_path)

    return {"error": 1. - acc_max, "cross_entropy": loss_min}
