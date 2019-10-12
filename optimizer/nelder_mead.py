import numpy as np
from optimizer.base_optimizer import BaseOptimizer
from utils import convert_hps_set, revert_hps, out_of_domain


def centroid(xs, ys):
    xs, ys = map(np.asarray, [xs, ys])
    order = np.argsort(ys)
    xs, ys = xs[order], ys[order]
    return xs[:-1].mean(axis=0)


class NelderMead(BaseOptimizer):
    def __init__(self,
                 hpu, 
                 n_parallels=1, 
                 n_init=10, 
                 max_evals=100,
                 delta_r=1.0,
                 delta_oc=0.5,
                 delta_ic=-0.5,
                 delta_e=2.0,
                 delta_s=0.5):

        super().__init__(hpu, rs=False, n_parallels=n_parallels, n_init=n_init, max_evals=max_evals)
        self.delta = {"r": delta_r,
                      "e": delta_e,
                      "s": delta_s,
                      "ic": delta_ic,
                      "oc": delta_oc}
        self.opt = self.sample()

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
        n_dim = len(cs._hyperparameters)
        Xc = convert_hps_set(X, cs)
        xs = Xc[:n_dim+1]
        ys = Y[:n_dim+1]
        n_evals = len(Y)
        idx = n_dim + 1

        while True:
            xr = self.reflect(xs, ys)
            if idx == n_evals:
                return revert_hps(xr, cs)
            else:
                yr = Y[idx]
                idx += 1

            if ys[0] <= yr < ys[-2]:
                xs[-1] = xr
                ys[-1] = yr

            elif yr < ys[0]:
                xe = self.expand(xs, ys)
                if idx == n_evals:
                    return revert_hps(xr, cs)
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
                    return revert_hps(xoc, cs)
                else:
                    yoc = Y[idx]
                    idx += 1
                    if yoc <= yr:
                        xs[-1] = xoc
                        ys[-1] = yoc
                    else:
                        if n_evals > idx + n_dim:
                            self.shrink(xs, ys, Y[idx:])
                        else:
                            this_idx = n_evals - idx
                            return_x = xs[this_idx] + self.delta["s"] * (xs[this_idx] - xs[0])
                            return revert_hps(return_x, cs)
            elif ys[-1] <= yr:
                xic = self.inside_contract(xs, ys)
                if idx == n_evals:
                    return revert_hps(xic, cs)
                else:
                    yic = Y[idx]
                    idx += 1
                    if yic < ys[-1]:
                        xs[-1] = xic
                        ys[-1] = yic
                    else:
                        if n_evals > idx + n_dim:
                            self.shrink(xs, ys, Y[idx:])
                        else:
                            this_idx = n_evals - idx
                            return_x = xs[this_idx] + self.delta["s"] * (xs[this_idx] - xs[0])
                            return revert_hps(return_x, cs)

    def sample(self, X, Y, hpu, job_id, lock):
        cs = hpu.config_space

        x = self.search(X, Y, cs)
        if out_of_domain(x, hpu):
            hpu.save_hps(x, [], job_id, lock, converted=False)
            X.append(x)
            Y.append(1.0e+8)
        else:
            return x
