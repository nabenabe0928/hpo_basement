import numpy as np
from scipy.special import erf


EPS = 1.0e-12


class GaussKernel():
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = np.maximum(sigma, EPS)

    def pdf(self, x):
        z = np.sqrt(2 * np.pi) * self.sigma
        mahalanobis = ((x - self.mu) / self.sigma) ** 2
        return 1. / z * np.exp(-0.5 * mahalanobis)

    def cdf(self, x):
        z = (x - self.mu) / (np.sqrt(2) * self.sigma)
        return 0.5 * (1. + erf(z))

    def sample_from_kernel(self, rng):
        return rng.normal(loc=self.mu, scale=self.sigma)


class AitchisonAitkenKernel():
    def __init__(self, choice, n_choices, top=0.8):
        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError("The choice must be between {} and {}, but {} was given.".format(0, self.n_choices - 1, x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))
