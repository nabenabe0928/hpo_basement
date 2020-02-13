import numpy as np
import torch
from optimizer.base_optimizer import BaseOptimizer
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import joint_optimize


def optimize_EI(gp, best_f, n_dim):
    """
    Reference: https://botorch.org/api/optim.html

    bounds: 2d-ndarray (2, D)
        The values of lower and upper bound of each parameter.
    q: int
        The number of candidates to sample
    num_restarts: int
        The number of starting points for multistart optimization.
    raw_samples: int
        The number of initial points.

    Returns for joint_optimize is (num_restarts, q, D)
    """

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    ei = ExpectedImprovement(gp, best_f=best_f, maximize=False)
    bounds = torch.from_numpy(np.array([[0.] * n_dim, [1.] * n_dim]))
    x = joint_optimize(ei,
                       bounds=bounds,
                       q=1,
                       num_restarts=3,
                       raw_samples=15)

    return np.array(x[0])


class SingleTaskGPBO(BaseOptimizer):
    def __init__(self, hp_utils, opt_requirements, experimental_settings, obj=None):

        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        self.opt = self.sample
        self.n_dim = len(self.hp_utils.config_space._hyperparameters)

    def sample(self):
        """
        Training Data: ndarray (N, D)
        Training Label: ndarray (N, )
        """

        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, y = map(np.asarray, [X, Y[0]])
        y = (y - y.mean()) / y.std()
        X, y = torch.from_numpy(X), torch.from_numpy(y)

        gp = SingleTaskGP(X, y)
        x = optimize_EI(gp, y.min(), self.n_dim)

        return self.hp_utils.revert_hp_conf(x)


class MultiTaskGPBO(BaseOptimizer):
    def __init__(self, hp_utils, opt_requirements, experimental_settings, obj=None):

        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        transfer_info_paths = opt_requirements.transfer_info_paths
        self.opt = self.sample
        self.n_dim = len(self.hp_utils.config_space._hyperparameters)
        self.X, self.Y = self.hp_utils.load_transfer_hps_conf(transfer_info_paths, convert=True)
        self.Y = [[(yn - yn.mean()) / yn.std() for yn in Ym] for Ym in self.Y]

    def create_multi_task_X(self):
        _X, _y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        self.X[0], self.Y[0] = map(np.asarray, [_X, _y])
        self.Y[0] = [(yn - yn.mean()) / yn.std() for yn in self.Y[0]]

        Xc = []
        for i, Xi in enumerate(self.X):
            task_id = np.ones(len(Xi)) * i
            Xc.append(np.c_[Xi, task_id])

        X = Xc[0]
        Y = self.Y[0][0]
        for i in range(1, len(Xc)):
            X = np.r_[X, Xc[i]]
            Y = np.r_[Y, self.Y[i][0]]
        return X, Y

    def sample(self):
        """
        Here, N is the number of evaluations all over each task.

        Training Data: ndarray (N, D + 1)
            The last element is the index of tasks.

        Training Label: ndarray (N, )

        task_feature: int
            The index to obtain the task id. Generally, task_feature=D
        output_tasks: list of int
            The id of the target tasks.
        """

        X, Y = self.create_multi_task_X()
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        mtgp = MultiTaskGP(X, Y, task_feature=self.n_dim, output_tasks=[0])
        x = optimize_EI(mtgp, self.Y[0][0].min(), self.n_dim)

        return self.hp_utils.revert_hp_conf(x)
