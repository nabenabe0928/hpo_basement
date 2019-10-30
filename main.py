import utils
import optimizer


path = "history/log/"
transfer_info_pathes_part = ["SingleTaskGPBO/sphere_3d/000"]
transfer_info_pathes = [path + transfer_info_path_part for transfer_info_path_part in transfer_info_pathes_part]

if __name__ == '__main__':

    opt_requirements, experimental_settings = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities(experimental_settings)
    # opt = optimizer.RandomSearch(hp_utils, opt_requirements, experimental_settings)
    opt = optimizer.SingleTaskMultivariateTPE(hp_utils, opt_requirements, experimental_settings)
    best_conf, best_performance = opt.optimize()
