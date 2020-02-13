import numpy as np
from optimizer.robo import models, acquisition_functions, maximizers
from optimizer.base_optimizer import BaseOptimizer


class SingleTaskBOHAMIANN(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 opt_requirements,
                 experimental_settings,
                 obj=None):
        """
        lower: ndarray (D, )
            The lower bound of each parameter
        upper: ndarray (D, )
            The upper bound of each parameter
        """

        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        self.opt = self.sample
        self.n_dim = len(self.hp_utils.config_space._hyperparameters)
        self.lower = np.zeros(self.n_dim)
        self.upper = np.ones(self.n_dim)
        self.model_objective = models.WrapperBohamiann()
        self.acquisition_func = acquisition_functions.LogEI(self.model_objective)
        self.maximizer = maximizers.DifferentialEvolution(self.acquisition_func, self.lower, self.upper, rng=self.rng)

    def sample(self):
        """
        X: list of hp_confs (N, D)

        y: list of performance (yN, N)
            Y[0] is the performance used in optimization.
        """

        _X, _y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, y = map(np.asarray, [_X, _y])

        self.model_objective.train(X, y[0], do_optimize=True)
        self.acquisition_func.update(self.model_objective)

        x = self.maximizer.maximize()

        return self.hp_utils.revert_hp_conf(x)


class MultiTaskBOHAMIANN(BaseOptimizer):
    def __init__(self, hp_utils, opt_requirements, experimental_settings, obj=None):
        """
        n_tasks: int
            The number of types of tasks including the target task
        X: list of hp_confs for each task (M, Nm, D)
            Nm is the number of evaluated configurations of task M.
            The id of target task is 0.
        Y: list of performance for each task (M, yN, Nm)
            Y[:][0] is the performance used in optimization.
        """

        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        transfer_info_paths = opt_requirements.transfer_info_paths

        self.opt = self.sample
        self.n_tasks = len(transfer_info_paths) + 1
        self.n_dim = len(self.hp_utils.config_space._hyperparameters)
        self.X, self.Y = self.hp_utils.load_transfer_hps_conf(transfer_info_paths, convert=True)
        self.lower = np.zeros(self.n_dim)
        self.upper = np.ones(self.n_dim)
        self.model_objective = models.WrapperBohamiannMultiTask(n_tasks=self.n_tasks)
        self.acquisition_func = acquisition_functions.LogEI(self.model_objective)

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
        """
        Reference: https://github.com/automl/RoBO/blob/master/robo/fmin/warmstart_mtbo.py

        Here, N is the number of evaluations all over each task.

        Training Data: ndarray (N, D + 1)
            The last element is the index of tasks.

        Training Label: ndarray (N, )
        """

        X, y = self.create_multitask_X()
        y_star = self.Y[0][0].min()

        # Optimize acquisition function only on the main task
        def wrapper(x):
            x_ = np.append(x, np.zeros([x.shape[0], 1]), axis=1)
            a = self.acquisition_func(x_, eta=y_star)
            return a

        maximizer = maximizers.DifferentialEvolution(wrapper, self.lower, self.upper, rng=self.rng)
        self.model_objective.train(X, y, do_optimize=True)
        self.acquisition_func.update(self.model_objective)
        x = maximizer.maximize()

        return self.hp_utils.revert_hp_conf(x)
