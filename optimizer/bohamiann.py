import numpy as np
import torch
from optimizer.base_optimizer import BaseOptimizer
from robo.fmin.bayesian_optimization import bayesian_optimization


class SingleTaskBOHAMIANN(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
                 verbose=True,
                 print_freq=1
                 ):

        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments,
                         restart=restart,
                         seed=seed,
                         verbose=verbose,
                         print_freq=print_freq
                         )
        self.opt = self.sample
        self.n_dim = len(hp_utils.config_space._hyperparameters)

    def sample(self):
        lower = np.zeros(self.n_dim)
        upper = np.ones(self.n_dim)
        results = bayesian_optimization(objective_function,
                                        lower,
                                        upper,
                                        model_type="bohamiann",
                                        num_iterations=0)

        return self.hp_utils.revert_hp_conf(x)


class MultiTaskBOHAMIANN(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
                 transfer_info_pathes=None
                 ):

        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments,
                         restart=restart,
                         seed=seed
                         )
        self.opt = self.sample
        self.X, self.Y = hp_utils.load_transfer_hps_conf(transfer_info_pathes, convert=True)

    def sample(self):
        _X, _Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        self.X[0], self.Y[0] = map(np.asarray, [_X, _Y])

        Xc = []
        for i, Xi in enumerate(self.X):
            task_id = np.ones(len(Xi)) * i
            Xc.append(np.c_[Xi, task_id])

        X = Xc[0]
        Y = self.Y[0][0]
        for i in range(1, len(Xc)):
            X = np.r_[X, Xc[i]]
            Y = np.r_[Y, self.Y[i][0]]

        warmstart_mtbo(objective_function,
                       lower,
                       upper,
                       observed_X,
                       observed_Y,
                       n_tasks=2,
                       num_iterations=30,
                       model_type="bohamiann",
                       target_task_id=1,
                       burnin
                       )

        return self.hp_utils.revert_hp_conf(x)
