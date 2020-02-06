from optimizer import BaseOptimizer
import utils
import numpy as np


class RandomSearch(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 opt_requirements,
                 experimental_settings,
                 obj=None,
                 given_default=None):
        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj, given_default=given_default)
        self.opt = self._initial_sampler


class LatinHypercubeSampling(BaseOptimizer):
    def __init__(self, hp_utils, opt_requirements, experimental_settings, obj=None):
        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        self.n_init = 0
        self.conf_idx = 0
        self.n_points = self.max_evals - self.n_jobs
        self.hp_confs = self.obtain_latin_hypercube()
        self.opt = self._initial_sampler

    def value_in_grid(self, rnd_grid, hp_idx, idx, n_conf, choices):
        if idx in hp_idx["numerical"]:
            i = hp_idx["numerical"].index(idx)
            return (rnd_grid[i][n_conf] - self.rng.uniform()) / self.n_points
        else:
            i = hp_idx["categorical"].index(idx)
            n_choice = len(choices[i])
            return self.rng.randint(n_choice)

    def obtain_latin_hypercube(self):
        hps = self.hp_utils.config_space._hyperparameters
        hp_idx = {"numerical": [], "categorical": []}
        choices = []

        for var_name, hp in hps.items():
            idx = self.hp_utils.config_space._hyperparameter_idx[var_name]
            dist = utils.distribution_type(self.hp_utils.config_space, var_name)
            if dist is str or dist is bool:
                hp_idx["categorical"].append(idx)
                choices.append(hp.choices)
            else:
                hp_idx["numerical"].append(idx)

        n_num_dim = len(hp_idx["numerical"])
        n_dim = n_num_dim + len(hp_idx["categorical"])
        rnd_grid = np.array([self.rng.permutation(list(range(1, self.n_points + 1))) for _ in range(n_num_dim)])

        hp_confs = [[self.value_in_grid(rnd_grid, hp_idx, i, k, choices) for i in range(n_dim)]
                    for k in range(self.n_points)]
        hp_confs = self.hp_utils.revert_hp_confs(hp_confs)

        return hp_confs

    def sample(self):
        self.conf_idx += 1
        return self.hp_confs[self.conf_idx - 1]
