import numpy as np
import matplotlib.pyplot as plt
from optimizer.parzen_estimator import GaussKernel, AitchisonAitkenKernel


EPS = 1e-12


def plot_density_estimators(pe_lower, pe_upper, var_name, pr_basis=False, pr_ei=False, pr_basis_mu=False):
    weights_set = [pe_lower.weights, pe_upper.weights]
    basis_set = [pe_lower.basis, pe_upper.basis]
    mus_set = [pe_lower.mus, pe_upper.mus]
    names = ["lower", "upper"]
    lb, ub = pe_lower.lb, pe_lower.ub
    cmap = plt.get_cmap("tab10")

    x = np.linspace(lb, ub, 100)
    des = np.array([np.zeros(100) for _ in range(2)])

    for i, (weights, basis, mus, de, name) in enumerate(zip(weights_set, basis_set, mus_set, des, names)):
        for w, b, mu in zip(weights, basis, mus):
            de += w * b.pdf(x)
            if pr_basis:
                plt.plot(x, w * b.pdf(x), color=cmap(i), linestyle="dotted")
            if pr_basis_mu:
                plt.plot([mu] * 100, np.linspace(0, w, 100), color=cmap(i), linestyle="dotted")
        plt.plot(x, de, label=name, color=cmap(i))

    if pr_ei:
        plt.plot(x, np.log(des[0]) - np.log(des[1]), label="EI function", color=cmap(2))

    plt.title("Parzen Estimators for {} with {} lower and {} upper evaluations.".format(var_name, len(mus_set[0]) - 1, len(mus_set[1]) - 1))
    plt.xlim(lb, ub)
    plt.grid()
    plt.legend()
    plt.show()


class NumericalParzenEstimator(object):
    def __init__(self, samples, lb, ub, weight_func, q=None):
        weights, mus, sigmas = self._calculate(samples, lb, ub, weight_func)
        self.weights, self.mus, self.sigmas = map(np.asarray, (weights, mus, sigmas))
        self.basis = [GaussKernel(m, s) for m, s in zip(mus, sigmas)]
        self.lb, self.ub, self.q = lb, ub, q

    def sample_from_density_estimator(self, rng, n_samples):
        samples = np.asarray([], dtype=float)
        while samples.size < n_samples:
            active = np.argmax(rng.multinomial(1, self.weights))
            drawn_hp = self.basis[active].sample_from_kernel(rng)
            if self.lb <= drawn_hp <= self.ub:
                samples = np.append(samples, drawn_hp)

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def log_likelihood(self, samples):
        p_accept = np.sum([w * (b.cdf(self.ub) - b.cdf(self.lb)) for w, b in zip(self.weights, self.basis)])
        ps = np.zeros(samples.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            if self.q is None:
                ps += w * b.pdf(samples)
            else:
                integral_u = b.cdf(np.minimum(samples + 0.5 * self.q, self.ub))
                integral_l = b.cdf(np.maximum(samples + 0.5 * self.q, self.lb))
                ps += w * (integral_u - integral_l)
        return np.log(ps + EPS) - np.log(p_accept + EPS)

    def _calculate_mus(self, samples, lower_bound, upper_bound):
        order = np.argsort(samples)
        sorted_mus = samples[order]
        prior_mu = 0.5 * (lower_bound + upper_bound)
        prior_pos = np.searchsorted(samples[order], prior_mu)
        sorted_mus = np.insert(sorted_mus, prior_pos, prior_mu)

        return sorted_mus, order, prior_pos

    def _calculate_sigmas(self, samples, lower_bound, upper_bound, sorted_mus, prior_pos):
        sorted_mus_with_bounds = np.insert([lower_bound, upper_bound], 1, sorted_mus)
        sigma = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2], sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigma[0] = sorted_mus_with_bounds[2] - sorted_mus_with_bounds[1]
        sigma[-1] = sorted_mus_with_bounds[-2] - sorted_mus_with_bounds[-3]

        maxsigma = upper_bound - lower_bound
        minsigma = (upper_bound - lower_bound) / min(100.0, (1.0 + len(sorted_mus)))
        sigma = np.clip(sigma, minsigma, maxsigma)

        prior_sigma = 1.0 * (upper_bound - lower_bound)
        sigma[prior_pos] = prior_sigma

        return sigma

    def _calculate_weights(self, samples, weights_func, order, prior_pos):
        sorted_weights = weights_func(samples.size)[order]
        sorted_weights = np.insert(sorted_weights, prior_pos, 1.)
        sorted_weights /= sorted_weights.sum()

        return sorted_weights

    def _calculate(self, samples, lower_bound, upper_bound, weights_func):
        samples = np.asarray(samples)
        sorted_mus, order, prior_pos = self._calculate_mus(samples, lower_bound, upper_bound)
        sigma = self._calculate_sigmas(samples, lower_bound, upper_bound, sorted_mus, prior_pos)
        sorted_weights = self._calculate_weights(samples, weights_func, order, prior_pos)

        return np.array(sorted_weights), np.array(sorted_mus), np.array(sigma)


class CategoricalParzenEstimator():
    def __init__(self, samples, n_choices, weights_func, top=0.8):
        self.n_choices = n_choices
        self.mus = samples
        self.basis = [AitchisonAitkenKernel(c, n_choices, top=top) for c in samples]
        self.weights = weights_func(len(samples))
        self.weights /= self.weights.sum()

    def sample_from_density_estimator(self, rng, n_samples):
        basis_samples = rng.multinomial(n=1, pvals=self.weights, size=n_samples)
        basis_idxs = np.dot(basis_samples, np.arange(self.weights.size))
        return np.array([self.basis[idx].sample_from_kernel(rng) for idx in basis_idxs])

    def log_likelihood(self, samples):
        ps = np.zeros(samples.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.cdf_for_numpy(samples)
        return np.log(ps + EPS)
