import utils
import optimizer


if __name__ == '__main__':
    requirements, experimental_settings = utils.parse_requirements()
    func_name = "styblinski"
    # experimental_settings["biased_cls"] = []
    path = "history/log/"
    transfer_info_pathes_part = ["SingleTaskGPBO/sphere_3d/000"]
    transfer_info_pathes = [path + transfer_info_path_part for transfer_info_path_part in transfer_info_pathes_part]

    hp_utils = utils.HyperparameterUtilities(func_name, experimental_settings=experimental_settings)
    # opt = optimizer.SingleTaskGPBO(hp_utils, **requirements)
    opt = optimizer.SingleTaskTPE(hp_utils, **requirements)
    # opt = optimizer.RandomSearch(hp_utils, **requirements)
    # opt = optimizer.MultiTaskGPBO(hp_utils, **requirements, transfer_info_pathes=transfer_info_pathes)
    best_conf, best_performance = opt.optimize()
