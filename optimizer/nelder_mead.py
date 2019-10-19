import numpy as np
from optimizer.base_optimizer import BaseOptimizer


class NelderMead(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 restart=True,
                 seed=None,
                 verbose=True,
                 print_freq=1,
                 delta_r=1.0,
                 delta_oc=0.5,
                 delta_ic=-0.5,
                 delta_e=2.0,
                 delta_s=0.5):

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
        self.delta = {"r": delta_r,
                      "e": delta_e,
                      "s": delta_s,
                      "ic": delta_ic,
                      "oc": delta_oc}
        self.opt = self.sample
        self.n_dim = n_init - 1
        self.idx = 0
        self.n_evals = 0
        self.xc = None
        self.xs = None
        self.ys = None

    def centroid(self):
        order = np.argsort(self.ys)
        self.xs, self.ys = self.xs[order], self.ys[order]
        self.xc = self.xs[:-1].mean(axis=0)

    def reflect(self):
        return self.xc + self.delta["r"] * (self.xc - self.xs[-1])

    def outside_contract(self):
        return self.xc + self.delta["oc"] * (self.xc - self.xs[-1])

    def if_outside_contract(self, X, Y):
        if Y[self.idx] <= Y[self.idx - 1]:
            self.xs[-1] = X[self.idx][:]
            self.ys[-1] = Y[self.idx]
            self.idx += 1
            return None
        else:
            self.idx += 1
            return self.if_shrink(Y)

    def inside_contract(self):
        return self.xc + self.delta["ic"] * (self.xc - self.xs[-1])

    def if_inside_contract(self, X, Y):
        if Y[self.idx] < self.ys[-1]:
            self.xs[-1] = X[self.idx][:]
            self.ys[-1] = Y[self.idx]
            self.idx += 1
            return None
        else:
            self.idx += 1
            return self.if_shrink(Y)

    def expand(self):
        return self.xc + self.delta["e"] * (self.xc - self.xs[-1])

    def if_expand(self, X, Y):
        if Y[self.idx] < Y[self.idx - 1]:
            self.xs[-1] = X[self.idx][:]
            self.ys[-1] = Y[self.idx]
        else:
            self.xs[-1] = X[self.idx - 1][:]
            self.ys[-1] = Y[self.idx - 1]
        self.idx += 1

    def shrink(self, Y):
        n_points = len(self.ys)
        for i in range(1, n_points):
            self.xs[i] = self.xs[0] + self.delta["s"] * (self.xs[i] - self.xs[0])
            self.ys[i] = Y[self.idx + i - 1]

    def if_shrink(self, Y):
        n_res = self.n_evals - self.idx
        if n_res >= self.n_dim:
            self.shrink(Y)
            self.idx += self.n_dim
            return None
        else:
            return self.xs[n_res + 1] + self.delta["s"] * (self.xs[n_res + 1] - self.xs[0])

    def search(self, X, Y):
        self.xs = X[:self.n_init]
        self.ys = Y[:self.n_init]
        self.n_evals = len(Y)
        self.idx = self.n_init

        while True:
            self.centroid()
            if self.idx == self.n_evals:
                return self.reflect()
            xr, yr = X[self.idx][:], Y[self.idx]
            self.idx += 1
            if self.ys[0] <= yr < self.ys[-2]:
                self.xs[-1], self.ys[-1] = xr, yr
            elif yr < self.ys[0]:
                if self.idx == self.n_evals:
                    return self.expand()
                else:
                    self.if_expand(X, Y)
            else:
                if self.idx == self.n_evals:
                    return self.inside_contract() if self.ys[-1] <= yr else self.outside_contract()
                else:
                    return_x = self.if_inside_contract(X, Y) if self.ys[-1] <= yr else self.if_outside_contract(X, Y)
                    if return_x is not None:
                        return return_x

    def sample(self):
        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)
        X, Y = map(np.asarray, [X, Y[0]])

        x = self.search(X, Y)
        return self.hp_utils.revert_hp_conf(x)
