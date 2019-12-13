import utils
import optimizer
from obj_functions.machine_learning_utils import start_train


path = "history/log/"
transfer_info_pathes_part = ["SingleTaskGPBO/sphere_3d/000"]
transfer_info_pathes = [path + transfer_info_path_part for transfer_info_path_part in transfer_info_pathes_part]


def objective_function(hp_conf, hp_utils, cuda_id, job_id, verbose=True, print_freq=1, save_time=None):
    from obj_functions import models, datasets

    hp_dict = hp_utils.list_to_dict(hp_conf)
    train_dataset, test_dataset = datasets.get_dataset(experimental_settings)
    save_path = "history/stdo" + hp_utils.save_path[11:] + "/log{:0>5}.csv".format(job_id)

    model = models.CNN(**hp_dict, n_cls=experimental_settings.n_cls)
    train_data, test_data = datasets.get_data(train_dataset, test_dataset, batch_size=model.batch_size)
    print(hp_dict)
    import torch
    device = torch.device("cuda", cuda_id)
    model = model.to(device)
    loss_min, acc_max = start_train(model, train_data, test_data, cuda_id, save_path)
    return {"error": 1. - acc_max, "cross_entropy": loss_min}


if __name__ == '__main__':
    """
    import numpy as np
    k = 2
    n_trial = 10
    gamma = []
    gamma.append(lambda x: min(int(np.ceil(0.1 * x)), 26))
    gamma.append(lambda x: min(int(np.ceil(0.125 * x)), 26))
    gamma.append(lambda x: min(int(np.ceil(0.15 * x)), 26))
    gamma.append(lambda x: min(int(np.ceil(0.175 * x)), 26))
    gamma.append(lambda x: min(int(np.ceil(0.2 * x)), 26))
    gamma.append(lambda x: min(int(np.ceil(0.25 * np.sqrt(x))), 26))
    """

    opt_requirements, experimental_settings = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities(experimental_settings)
    # opt = optimizer.SingleTaskUnivariateTPE(hp_utils, opt_requirements, experimental_settings, gamma_func=gamma[k])
    # opt = optimizer.SingleTaskUnivariateTPE(hp_utils, opt_requirements, experimental_settings)
    # opt = optimizer.NelderMead(hp_utils, opt_requirements, experimental_settings)
    opt = optimizer.RandomSearch(hp_utils, opt_requirements, experimental_settings, obj=objective_function)
    best_conf, best_performance = opt.optimize()
