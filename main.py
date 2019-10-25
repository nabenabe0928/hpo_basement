import utils
import optimizer


if __name__ == '__main__':
    requirements, experimental_settings = utils.parse_requirements()
    func_name = "griewank"
    # experimental_settings["biased_cls"] = []
    # transfer_info_pathes = []
    
    hp_utils = utils.HyperparameterUtilities(func_name, experimental_settings=experimental_settings)
    opt = optimizer.SingleTaskTPE(hp_utils, **requirements)
    # opt = optimizer.RandomSearch(hp_utils, **requirements)
    # opt = optimizer.NelderMead(hp_utils, **requirements)
    # opt = optimizer.SingleTaskGPBO(hp_utils, **requirements)
    # opt = optimizer.MultiTaskGPBO(hp_utils, **requirements, transfer_info_pathes=transfer_info_pathes)
    best_conf, best_performance = opt.optimize()
