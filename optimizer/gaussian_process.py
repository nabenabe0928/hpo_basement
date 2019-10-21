import numpy as np
import torch
from optimizer.base_optimizer import BaseOptimizer
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import joint_optimize


def optimize_EI(gp, best_f, n_dim):
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    ei = ExpectedImprovement(gp, best_f=best_f, maximize=False)
    bounds = torch.from_numpy(np.array([[0.] * n_dim, [1.] * n_dim]))
    x = joint_optimize(ei,
                       bounds=bounds,
                       q=1,
                       num_restarts=5,
                       raw_samples=5)

    return np.array(x[0])


class SingleTaskGPBO(BaseOptimizer):
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
        """
        X: N * D
        """
        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, Y = map(np.asarray, [X, Y[0]])
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)

        gp = SingleTaskGP(X, Y)
        x = optimize_EI(gp, Y[0].min(), self.n_dim)

        return self.hp_utils.revert_hp_conf(x)


class MultiTaskGPBO(BaseOptimizer):
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
        self.n_dim = len(hp_utils.config_space._hyperparameters)
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

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        mtgp = MultiTaskGP(X, Y, task_feature=self.n_dim, output_tasks=0)
        x = optimize_EI(mtgp, self.Y[0][0].min(), self.n_dim)

        return self.hp_utils.revert_hp_conf(x)
