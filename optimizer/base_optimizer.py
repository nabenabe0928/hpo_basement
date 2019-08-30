import numpy as np
from utils import import_module
from hp_utils import distribution_type


class BaseOptimizer():
    def __init__(self, opt_name, obj_name, max_evals, n_init, hpu, **kwargs):
        self.n_evals = 0

        self.hpu = hpu
        self.n_init = n_init
        self.opt = import_module(opt_name)

    def _initial_sampler(self):
        cs = self.hpu.config_space
        hps = cs._hyperparameters
        sample = [None for _ in range(len(hps))]

        for var_name, hp in hps.items():
            idx = cs._hyperparameter_idx[var_name]
            dist = distribution_type(cs, var_name)
            if dist == str:
                # categorical
                choices = hp.choices
                sample[idx] = np.random.randint(len(choices))
            else:
                # numerical
                l, u, q, log = hp.lower, hp.upper, hp.q, hp.log
                if log is not None:
                    if q is not None:
                        raise ValueError("q and log cannot be not None at the same time.")
                    l, u = np.log(l), np.log(u)
                    sample[idx] = dist(np.exp(np.random.random() * (u - l) + l))
                elif q is not None:
                    sample[idx] = dist(np.floor((np.random.random() * (u - l) + l) / q) * q)
                else:
                    sample[idx] = dist(np.random.random() * (u - l) + l)

        self.n_evals += 1

        return sample

    def start_opt(self, obj):
        for it in range(self.max_evals):
            if it < self.n_init:
                sample = self._initial_sampler()
                obj(sample)
            else:
                sample = self.opt()