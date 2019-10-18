import numpy as np
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
