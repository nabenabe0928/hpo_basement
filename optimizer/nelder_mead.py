import numpy as np
from optimizer.base_optimizer import BaseOptimizer
import utils


def centroid(xs, ys):
    xs, ys = map(np.asarray, [xs, ys])
    order = np.argsort(ys)
    xs, ys = xs[order], ys[order]
    return xs[:-1].mean(axis=0)


class NelderMead(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 max_evals=100,
                 n_experiments=0,
                 delta_r=1.0,
                 delta_oc=0.5,
                 delta_ic=-0.5,
                 delta_e=2.0,
                 delta_s=0.5):

        super().__init__(hp_utils,
                         n_parallels=n_parallels,
                         n_init=n_init,
                         max_evals=max_evals,
                         n_experiments=n_experiments
                         )
        self.delta = {"r": delta_r,
                      "e": delta_e,
                      "s": delta_s,
                      "ic": delta_ic,
                      "oc": delta_oc}
        self.opt = self.sample
        self.n_dim = n_init - 1

    def reflect(self, xs, ys):
        xc = centroid(xs, ys)
        return xc + self.delta["r"] * (xs - xs[-1])

    def outside_contract(self, xs, ys):
        xc = centroid(xs, ys)
        return xc + self.delta["oc"] * (xs - xs[-1])

    def inside_contract(self, xs, ys):
        xc = centroid(xs, ys)
        return xc + self.delta["ic"] * (xs - xs[-1])

    def expand(self, xs, ys):
        xc = centroid(xs, ys)
        return xc + self.delta["e"] * (xs - xs[-1])

    def shrink(self, xs, ys, Yp):
        n_points = len(ys)
        for i in range(1, n_points):
            xs[i] = xs[0] + self.delta["s"] * (xs[i] - xs[0])
            ys[i] = Yp[i - 1]

    def search(self, X, Y, cs):
        xs = X[:self.n_init]
        ys = Y[:self.n_init]
        n_evals = len(Y)
        idx = self.n_init

        while True:
            xr = self.reflect(xs, ys)
            if idx == n_evals:
                return xr
            else:
                yr = Y[idx]
                idx += 1
            if ys[0] <= yr < ys[-2]:
                xs[-1] = xr
                ys[-1] = yr

            elif yr < ys[0]:
                xe = self.expand(xs, ys)
                if idx == n_evals:
                    return xe
                else:
                    ye = Y[idx]
                    idx += 1
                    if ye < yr:
                        xs[-1] = xe
                        ys[-1] = ye
                    else:
                        xs[-1] = xr
                        ys[-1] = yr

            elif ys[-2] <= yr < ys[-1]:
                xoc = self.outside_contract(xs, ys)
                if idx == n_evals:
                    return xoc
                else:
                    yoc = Y[idx]
                    idx += 1
                    if yoc <= yr:
                        xs[-1] = xoc
                        ys[-1] = yoc
                    else:
                        if n_evals > idx + self.n_dim:
                            self.shrink(xs, ys, Y[idx:])
                            idx += self.n_dim
                        else:
                            this_idx = n_evals - idx
                            return_x = xs[this_idx] + self.delta["s"] * (xs[this_idx] - xs[0])
                            return return_x
            elif ys[-1] <= yr:
                xic = self.inside_contract(xs, ys)
                if idx == n_evals:
                    return xic
                else:
                    yic = Y[idx]
                    idx += 1
                    if yic < ys[-1]:
                        xs[-1] = xic
                        ys[-1] = yic
                    else:
                        if n_evals > idx + self.n_dim:
                            self.shrink(xs, ys, Y[idx:])
                            idx += self.n_dim
                        else:
                            this_idx = n_evals - idx
                            return_x = xs[this_idx] + self.delta["s"] * (xs[this_idx] - xs[0])
                            return return_x

    def sample(self):
        cs = self.hp_utils.config_space
        X, Y = self.hp_utils.load_hps_conf(convert=True, do_sort=False)

        return utils.revert_hp_conf(self.search(X, Y[0], cs), cs)
