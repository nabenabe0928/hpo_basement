import numpy as np
import torch
from optimizer.base_optimizer import BaseOptimizer
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import joint_optimize


class SingleTaskGPBO(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
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
        self.idx = 0
        self.n_evals = 0

    def sample(self):
        """
        X: N * D
        """
        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, Y = map(np.asarray, [X, Y[0]])
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        ei = ExpectedImprovement(gp, best_f=Y.min(), maximize=False)
        x = joint_optimize(ei,
                           bounds=[[0.] * len(X[0]), [1.] * len(X[0])],
                           q=1,
                           num_restarts=5,
                           raw_samples=5)

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
        self.idx = 0
        self.n_evals = 0

        ### Must Fix here
        self.X, self.Y = None, None

    def sample(self):
        """
        X: N * D
        """

        self.X[0], self.Y[0] = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        self.X[0], self.Y[0] = map(np.asarray, [self.X[0], self.Y[0][0]])

        Xc, Yc = [], []
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

        gp = MultiTaskGP(X, Y, task_feature=len(self.X[0][0]), output_tasks=0)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        ei = ExpectedImprovement(gp, best_f=self.Y[0].min(), maximize=False)
        x = joint_optimize(ei,
                           bounds=[[0.] * len(X[0]), [1.] * len(X[0])],
                           q=1,
                           num_restarts=5,
                           raw_samples=5)

        return self.hp_utils.revert_hp_conf(x)
