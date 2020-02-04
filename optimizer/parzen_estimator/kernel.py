import numpy as np
from scipy.special import erf
from optimizer.constants import EPS, sq2, sq_pi


class GaussKernel():
    def __init__(self, mu, sigma, lb, ub, q):
        """
        The hyperparameters of Gauss Kernel.

        mu: float
            In general, this value is one of the observed values.
        sigma: float
            Generally, it is called band width and there are so many methods to initialize this value.
        lb, ub, q: float or int
            lower and upper bound and quantization value
        norm_const: float
            The normalization constant of probability density function.
            In other words, when we integranl this kernel from lb to ub, we would obtain 1 as a result.
        """

        self.mu = mu
        self.sigma = max(sigma, EPS)
        self.lb, self.ub, self.q = lb, ub, q
        self.norm_coef = 1. / sq2 / sq_pi / self.sigma
        self.norm_const = 1.
        self.norm_const = 1. / self.cdf_with_range(lb, ub)
        self.log_norm_coef = np.log(self.norm_coef * self.norm_const)

    def pdf(self, x):
        """
        Returning the value Probability density function of a given x.
        """

        if self.q is None:
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.norm_const * self.norm_coef * np.exp(-0.5 * mahalanobis)
        else:
            xl = np.maximum(x + 0.5 * self.q, self.lb)
            xu = np.minimum(x + 0.5 * self.q, self.ub)
            integral = self.cdf_with_range(xl, xu)
            return integral

    def log_pdf(self, x):
        if self.q is None:
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.log_norm_coef - 0.5 * mahalanobis
        else:
            xl = np.maximum(x + 0.5 * self.q, self.lb)
            xu = np.minimum(x + 0.5 * self.q, self.ub)
            integral = self.cdf_with_range(xl, xu)
            return np.log(integral)

    def cdf(self, x):
        """
        Returning the value of Cumulative distribution function at a given x.
        """

        z = (x - self.mu) / sq2 / self.sigma
        return np.maximum(self.norm_const * 0.5 * (1. + erf(z)), EPS)

    def cdf_with_range(self, lb, ub):
        zl = (lb - self.mu) / sq2 / self.sigma
        zu = (ub - self.mu) / sq2 / self.sigma
        return np.maximum(self.norm_const * 0.5 * (erf(zu) - erf(zl)), EPS)

    def sample_from_kernel(self, rng):
        """
        Returning the random number sampled from this Gauss kernel.
        """

        while True:
            sample = rng.normal(loc=self.mu, scale=self.sigma)
            if self.lb <= sample <= self.ub:
                return sample


class AitchisonAitkenKernel():
    def __init__(self, choice, n_choices, top=0.9):
        """
        Reference: http://www.ccsenet.org/journal/index.php/jmr/article/download/24994/15579

        Hyperparameters of Aitchison Aitken Kernel.

        n_choices: int
            The number of choices.
        choice: int
            The ID of the target choice.
        top: float (0. to 1.)
            The hyperparameter controling the extent of the other choice's distribution.
        """

        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        """
        Returning a probability of a given x.
        """

        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError("The choice must be between {} and {}, but {} was given.".format(0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        """
        Returning probabilities of a given list x.
        """

        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        """
        Returning probabilities of every possible choices.
        """

        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        """
        Returning random choice sampled from this Kernel.
        """

        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]


class UniformKernel():
    def __init__(self, n_choices):
        """
        Hyperparameters of Uniform Kernel.

        n_choices: int
            The number of choices.
        """

        self.n_choices = n_choices

    def cdf(self, x):
        """
        Returning a probability of a given x.
        """

        if 0 <= x <= self.n_choices - 1:
            return 1. / self.n_choices
        else:
            raise ValueError("The choice must be between {} and {}, but {} was given.".format(0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        """
        Returning probabilities of a given list x.
        """

        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        """
        Returning probabilities of every possible choices.
        """

        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        """
        Returning random choice sampled from this Kernel.
        """

        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]
