from optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(self, hp_utils, opt_requirements, experimental_settings, obj=None):
        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        self.opt = self._initial_sampler
