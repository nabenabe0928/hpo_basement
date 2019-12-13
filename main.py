import utils
import optimizer


path = "history/log/"
transfer_info_pathes_part = ["SingleTaskGPBO/sphere_3d/000"]
transfer_info_pathes = [path + transfer_info_path_part for transfer_info_path_part in transfer_info_pathes_part]


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
    opt = optimizer.SingleTaskUnivariateTPE(hp_utils, opt_requirements, experimental_settings)
    # opt = optimizer.NelderMead(hp_utils, opt_requirements, experimental_settings)
    best_conf, best_performance = opt.optimize()
