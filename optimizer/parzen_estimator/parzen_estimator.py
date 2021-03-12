import numpy as np
from optimizer.constants import EPS, sq2, sq_pi
from scipy.special import erf


class NumericalParzenEstimator():
    """
    samples: ndarray (n, )
        The observed hyperparameter values.
    lb: float
        The lower bound of a hyperparameter
    ub: float
        The upper bound of a hyperparameter
    weight_func: callable
        The function returning the weight of each basis of this parzen estimator.
    q: float
        The quantization value.
    """

    def __init__(self, samples, lb, ub, weight_func, q=None, rule="james", prior=True):
        """
        Here, the number of basis is n + 1.
        n basis are from observed values and 1 basis is from the prior distribution
        which is N((lb + ub) / 2, (ub - lb) ** 2).

        weights: ndarray (n + 1, )
            the weight of each basis. The total must be 1.
        mus: ndarray (n + 1, )
            The center of each basis.
            The values themselves are the observed hyperparameter values. Sorted in ascending order.
        sigmas: ndarray (n + 1, )
            The band width of each basis.
            The values are determined by a heuristic.
        sq2_sigmas: ndarray (n + 1, )
            Define to reduce the computational time.
            It is equal to np.sqrt(2) * sigmas
        normal_terms: ndarray (n + 1, )
            Define to reduce the computational time.
            The normalization coefficient for each kernel.
        gauss_coefs: ndarray (n + 1, )
            Define to reduce the computational time.
            The product of normalization coefficient and coefficient put before the exponential part of gauss kernel.
        log_gauss_coefs: ndarray (n + 1, )
            Define to reduce the computational time.
            The log version of norm_consts
        """

        self.lb, self.ub, self.q, self.rule = lb, ub, q, rule
        self.weights, self.mus, self.sigmas = self._calculate(samples, weight_func, prior=prior)
        self.sq2_sigmas = sq2 * self.sigmas
        self.normal_terms = 1. / np.maximum(EPS, 0.5 * (erf((ub - self.mus) / self.sq2_sigmas)
                                                        - erf((lb - self.mus) / self.sq2_sigmas)))
        self.gauss_coefs = self.normal_terms / sq_pi / self.sq2_sigmas
        self.log_gauss_coefs = np.log(self.gauss_coefs)

    def sample_from_density_estimator(self, rng, n_samples):
        """
        Parameters
        ----------
        rng: numpy.random.RandomState object
        n_samples: int
            The number of samples

        Returns
        -------
        samples: ndarray (n_samples, )
            The random number sampled from the parzen estimator.
        """

        samples = np.array([
            self.sample_from_kernel(rng, np.argmax(rng.multinomial(1, self.weights)))
            for _ in range(n_samples)
        ])

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def sample_from_kernel(self, rng, idx):
        """
        Returning the random number sampled from a chosen Gauss kernel.
        """

        while True:
            sample = rng.normal(loc=self.mus[idx], scale=self.sigmas[idx])
            if self.lb <= sample <= self.ub:
                return sample

    def pdf(self, xs):
        """
        Parameters
        ----------
        xs: np.ndarray (n_ei_candidates, )
            The candidates for the next evaluation.

        Returns
        -------
        pdf: np.ndarray (n_ei_candidates, )
            The probablity density function values for each candidates.
        """
        if self.q is None:
            mahalanobis = ((xs[:, None] - self.mus) / self.sigmas) ** 2  # shape = (n_ei_candidates, n_basis)
            vals_for_each_basis_at_each_x = self.gauss_coefs * np.exp(-0.5 * mahalanobis)
            return vals_for_each_basis_at_each_x @ self.weights
        else:
            xl = np.maximum(xs + 0.5 * self.q, self.lb)
            xu = np.minimum(xs + 0.5 * self.q, self.ub)
            vals_for_each_basis_at_each_x = self.cdf_with_range(xl, xu)
            return vals_for_each_basis_at_each_x @ self.weights

    def log_likelihood(self, xs):
        """
        Parameters
        ----------
        xs: ndarray (n_ei_candidates, )
            The number of candidates to evaluate the EI function.

        Returns
        -------
        loglikelihood of the parzen estimator at given points: ndarray (n_ei_candidates, )
            Here, we do not consider jacobian, because it will be canceled out when we compute EI function.
        """

        return np.log(self.pdf(xs) + EPS)

    def basis_likelihood(self, xs):
        """
        Parameters
        ----------
        xs: np.ndarray (n_ei_candidates, )
            The candidates for the next evaluation.

        Returns
        -------
        pdf: np.ndarray (n_ei_candidates, n_basis)
            The probablity density function values of each basis at each candidate point.
        """
        if self.q is None:
            mahalanobis = ((xs[:, None] - self.mus) / self.sigmas) ** 2  # shape = (n_ei_candidates, n_basis)
            vals_for_each_basis_at_each_x = self.gauss_coefs * np.exp(-0.5 * mahalanobis)
        else:
            xl = np.maximum(xs + 0.5 * self.q, self.lb)
            xu = np.minimum(xs + 0.5 * self.q, self.ub)
            vals_for_each_basis_at_each_x = self.cdf_with_range(xl, xu)

        return vals_for_each_basis_at_each_x

    def basis_loglikelihood(self, xs):
        """
        Returns
        -------
        loglikelihood of each basis at given points: ndarray (n_ei_candidates, n_basis)
        """

        if self.q is None:
            mahalanobis = ((xs[:, None] - self.mus) / self.sigmas) ** 2
            return self.log_gauss_coefs - 0.5 * mahalanobis
        else:
            xl = np.maximum(xs + 0.5 * self.q, self.lb)
            xu = np.minimum(xs + 0.5 * self.q, self.ub)
            vals_for_each_basis_at_each_x = self.cdf_with_range(xl, xu)
            return np.log(vals_for_each_basis_at_each_x)

    def cdf_with_range(self, xl, xu):
        zl = (xl[:, None] - self.mus) / self.sq2_sigmas  # shape = (n_ei_candidates, n_basis)
        zu = (xu[:, None] - self.mus) / self.sq2_sigmas
        vals_for_each_basis_at_each_x = np.maxinum(self.normal_terms * 0.5 * (erf(zu) - erf(zl)), EPS)
        return vals_for_each_basis_at_each_x

    def _calculate(self, samples, weights_func, prior):
        if self.rule == "james":
            return self._calculate_by_james_rule(samples, weights_func, prior)
        elif self.rule == "scott":
            return self._calculate_by_scott_rule(samples, prior)
        else:
            raise ValueError("Rule must be 'scott' or 'james'.")

    def _calculate_by_james_rule(self, samples, weights_func, prior):
        mus = np.append(samples, 0.5 * (self.lb + self.ub)) if prior else np.array(samples)
        sigma_bounds = [(self.ub - self.lb) / min(100.0, mus.size), self.ub - self.lb]

        order = np.argsort(mus)
        sorted_mus = mus[order]
        original_order = np.argsort(order)
        prior_pos = np.where(order == mus.size - 1)[0][0]

        sorted_mus_with_bounds = np.insert([sorted_mus[0], sorted_mus[-1]], 1, sorted_mus)
        sigmas = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2],
                            sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigmas = np.clip(sigmas, sigma_bounds[0], sigma_bounds[1])

        if prior:
            sigmas[prior_pos] = sigma_bounds[1]

        weights = weights_func(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas[original_order]

    def _calculate_by_scott_rule(self, samples, prior):
        mus = np.append(samples, 0.5 * (self.lb + self.ub)) if prior else np.array(samples)
        mus_sigma = mus.std(ddof=1)
        IQR = np.subtract.reduce(np.percentile(mus, [75, 25]))
        sigma = 1.059 * min(IQR, mus_sigma) * mus.size ** (-0.2)
        sigmas = np.ones(mus.size) * np.clip(sigma, 1.0e-2 * (self.ub - self.lb), 0.5 * (self.ub - self.lb))
        if prior:
            sigmas[-1] = self.ub - self.lb
        weights = np.ones(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas


class CategoricalParzenEstimator():
    """
    Reference: http://www.ccsenet.org/journal/index.php/jmr/article/download/24994/15579

    Hyperparameters of Aitchison Aitken Kernel.

    n_choices: int
        The number of choices.
    choice: int
        The ID of the target choice.
    top: float (0. to 1.)
        The hyperparameter controling the extent of the other choice's distribution.
    basis_likelihoods, basis_loglikelihoods: ndarray (n_choices, n_basis)
        The likelihood value of a given basis at each choice and that of log value.
    likelihoods, loglikelihoods: ndarray (n_choices,)
        The likelihood value of each choice and that of log value.
    """
    def __init__(self, samples, n_choices, weights_func, top=0.8, prior=True):
        self.n_choices = n_choices
        self.samples = samples
        self.top = top
        self.n_basis = samples.size + prior
        self.weights = weights_func(self.n_basis)
        self.weights /= self.weights.sum()
        self.basis_likelihoods = None
        self.basis_loglikelihoods = None
        self.likelihoods = None
        self.loglikelihoods = None
        self._calculate_cdf(prior=prior)

    def _calculate_cdf(self, prior=True):
        bottom_val = (1. - self.top) / (self.n_choices - 1)
        self.likelihoods = np.ones(self.n_choices) * bottom_val
        self.basis_likelihoods = np.ones((self.n_choices, self.n_basis)) * bottom_val
        bottom_to_top = self.top - bottom_val

        for b, (c, w) in enumerate(zip(self.samples, self.weights)):
            if 0 <= c <= self.n_choices - 1:
                self.likelihoods[c] += w * bottom_to_top
                self.basis_likelihoods[c][b] = bottom_to_top
            else:
                raise ValueError(f"The choice must be between {0} and {self.n_choices - 1}, but {c} was given.")

        if prior:
            self.basis_likelihoods[:, -1] += 1. / self.n_choices - bottom_val
            self.likelihoods += self.weights[-1] * (1. / self.n_choices - bottom_val)
        print(self.likelihoods)
        self.basis_loglikelihoods = np.log(self.basis_likelihoods + EPS)
        self.loglikelihoods = np.log(self.likelihoods + EPS)

    def sample_from_density_estimator(self, rng, n_samples):
        n_samples_of_each_choice = rng.multinomial(n=1, pvals=self.likelihoods, size=n_samples)
        return np.dot(n_samples_of_each_choice, np.arange(self.n_choices))

    def pdf(self, xs):

        return self.likelihoods[xs]  # (n_ei_candidates,)

    def log_likelihood(self, xs):

        return self.loglikelihoods[xs]  # (n_ei_candidates,)

    def basis_loglikelihood(self, xs):

        return self.basis_loglikelihoods[xs]  # (n_ei_candidates, n_basis)

    def basis_likelihood(self, xs):

        return self.basis_likelihoods[xs]  # (n_ei_candidates, n_basis)
