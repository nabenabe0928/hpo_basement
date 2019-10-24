import numpy as np
from optimizer.robo.models.wrapper_bohamiann import WrapperBohamiannMultiTask, WrapperBohamiann
from optimizer.robo.acquisition_functions.log_ei import LogEI
from optimizer.robo.maximizers.differential_evolution import DifferentialEvolution
from optimizer.base_optimizer import BaseOptimizer


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
        self.lower = np.zeros(self.n_dim)
        self.upper = np.ones(self.n_dim)
        self.model_objective = WrapperBohamiann()
        self.acquisition_func = LogEI(self.model_objective)
        self.maximizer = DifferentialEvolution(self.acquisition_func, self.lower, self.upper)

    def sample(self):
        _X, _y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, y = map(np.asarray, [_X, _y])

        self.model_objective.train(X, y[0], do_optimize=True)
        self.acquisition_func.update(self.model_objective)

        x = self.maximizer.maximize()

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
        self.X, self.Y, self.n_tasks = hp_utils.load_transfer_hps_conf(transfer_info_pathes, convert=True)
        self.lower = np.zeros(self.n_dim)
        self.upper = np.ones(self.n_dim)
        self.model_objective = WrapperBohamiannMultiTask(n_tasks=self.n_tasks)
        self.acquisition_func = LogEI(self.model_objective)

    def create_multitask_X(self):
        _X, _y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        self.X[0], self.Y[0] = map(np.asarray, [_X, _y])

        Xc = []
        for i, Xi in enumerate(self.X):
            task_id = np.ones(len(Xi)) * i
            Xc.append(np.c_[Xi, task_id])

        X = Xc[0]
        y = self.Y[0][0]
        for i in range(1, len(Xc)):
            X = np.r_[X, Xc[i]]
            y = np.r_[y, self.Y[i][0]]

        return X, y

    def sample(self):
        X, y = self.create_multitask_X()
        y_star = self.Y[0][0].min()

        # Optimize acquisition function only on the main task
        def wrapper(x):
            x_ = np.append(x, np.zeros([x.shape[0], 1]), axis=1)
            a = self.acquisition_func(x_, eta=y_star)
            return a

        maximizer = DifferentialEvolution(wrapper, self.lower, self.upper)
        self.model_objective.train(X, y, do_optimize=True)
        self.acquisition_func.update(self.model_objective)
        x = maximizer.maximize()

        return self.hp_utils.revert_hp_conf(x)
