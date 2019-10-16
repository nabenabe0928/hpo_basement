import utils
import optimizer


if __name__ == '__main__':
    requirements, dim = utils.parse_requirements()
    hp_utils = utils.HyperparameterUtilities("sphere", dim=dim)
    # opt = optimizer.NelderMead(hp_utils, **requirements)
    opt = optimizer.RandomSearch(hp_utils, **requirements)
    opt.optimize()
