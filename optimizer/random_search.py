from optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None
                 ):
        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments,
                         restart=restart,
                         seed=seed
                         )
        self.opt = self._initial_sampler
