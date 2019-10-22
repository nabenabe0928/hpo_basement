import utils
import optimizer


if __name__ == '__main__':
    requirements, experimental_settings = utils.parse_requirements()
    # experimental_settings["biased_cls"] = []
    hp_utils = utils.HyperparameterUtilities("sphere", experimental_settings=experimental_settings)
    # opt = optimizer.NelderMead(hp_utils, **requirements)
    opt = optimizer.SingleTaskTPE(hp_utils, **requirements)
    opt.optimize()
