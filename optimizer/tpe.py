import numpy as np
from scipy.special import logsumexp
from optimizer.parzen_estimator import NumericalParzenEstimator, CategoricalParzenEstimator
from optimizer import BaseOptimizer


def default_gamma(x, n_samples_lower=26):

    return min(int(np.ceil(0.25 * np.sqrt(x))), n_samples_lower)


def default_weights(x, n_samples_lower=26):
    if x < n_samples_lower:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - n_samples_lower)
        flat = np.ones(n_samples_lower)
        return np.concatenate([ramp, flat], axis=0)


class BaseTPE(BaseOptimizer):
    def __init__(self,
                 hp_utils,
                 opt_requirements,
                 experimental_settings,
                 obj=None,
                 n_ei_candidates=24,
                 rule="james",
                 gamma_func=default_gamma,
                 weight_func=default_weights):
        """
        n_ei_candidates: int
            The number of points to evaluate the EI function.
        rule: str
            The rule of bandwidth selection.
        variate: str
            The dimensionality of parzen estimator.
            If "multi", using multivariate parzen estimator, elif "uni", using univariate one.
        gamma_func: callable
            The function returning the number of a better group based on the total number of evaluations.
        weight_func: callable
            The function returning the coefficients of each kernel.
        """

        super().__init__(hp_utils, opt_requirements, experimental_settings, obj=obj)
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weight_func = weight_func
        self.rule = rule

    def _construct_numerical_parzen_estimator(self, var_name, var_type, lower_hps, upper_hps):
        """
        Parameters
        ----------
        lower_hps: ndarray (N_lower, )
            The values of better group.
        upper_hps: ndarray (N_upper, )
            The values of worse group.
        var_name: str
            The name of a hyperparameter
        var_type: type
            The type of a hyperparameter
        """

        hp = self.hp_utils.config_space._hyperparameters[var_name]
        q, log, lb, ub, converted_q = hp.q, hp.log, 0., 1., None

        if not log and (var_type is int or q is not None):
            converted_q = 1. / (hp.upper - hp.lower) if q is None else q / (hp.upper - hp.lower)
            lb -= 0.5 * converted_q
            ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(lower_hps, lb, ub, self.weight_func, q=converted_q, rule=self.rule)
        pe_upper = NumericalParzenEstimator(upper_hps, lb, ub, self.weight_func, q=converted_q, rule=self.rule)

        return pe_lower, pe_upper

    def _construct_categorical_parzen_estimator(self, var_name, lower_hps, upper_hps):
        choices = self.hp_utils.config_space._hyperparameters[var_name].choices
        n_choices = len(choices)
        lower_hps = [choices.index(hp) for hp in lower_hps]
        upper_hps = [choices.index(hp) for hp in upper_hps]
        lower_hps, upper_hps = map(np.asarray, [lower_hps, upper_hps])

        pe_lower = CategoricalParzenEstimator(lower_hps, n_choices, self.weight_func)
        pe_upper = CategoricalParzenEstimator(upper_hps, n_choices, self.weight_func)

        return pe_lower, pe_upper, choices


class SingleTaskUnivariateTPE(BaseTPE):
    def __init__(self,
                 hp_utils,
                 opt_requirements,
                 experimental_settings,
                 obj=None,
                 n_ei_candidates=24,
                 rule="james",
                 gamma_func=default_gamma,
                 weight_func=default_weights):

        super().__init__(hp_utils,
                         opt_requirements,
                         experimental_settings,
                         obj=obj,
                         n_ei_candidates=n_ei_candidates,
                         rule=rule,
                         gamma_func=gamma_func,
                         weight_func=weight_func)

        self.opt = self.sample

    def sample(self):
        hps_conf, _ = self.hp_utils.load_hps_conf(convert=True, do_sort=True, index_from_conf=False)
        hp_conf = []

        for var_name, hps in zip(self.hp_utils.var_names, hps_conf):
            n_lower = self.gamma_func(len(hps))
            lower_hps, upper_hps = hps[:n_lower], hps[n_lower:]
            var_type = self.hp_utils.dist_types[var_name]

            if var_type in [float, int]:
                pe_lower, pe_upper = self._construct_numerical_parzen_estimator(var_name, var_type, lower_hps, upper_hps)
                hp_value = self._compare_candidates(pe_lower, pe_upper)
            else:
                pe_lower, pe_upper, choices = self._construct_categorical_parzen_estimator(var_name, lower_hps, upper_hps)
                hp_value = self._compare_candidates(pe_lower, pe_upper, choices)
            hp_conf.append(hp_value)

        return self.hp_utils.revert_hp_conf(hp_conf)

    def _compare_candidates(self, pe_lower, pe_upper, choices=None):
        samples_lower = pe_lower.sample_from_density_estimator(self.rng, self.n_ei_candidates)
        best_idx = np.argmax(pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower))
        if choices is None:
            return samples_lower[best_idx]
        else:
            best_choice_idx = int(samples_lower[best_idx])
            return choices[best_choice_idx]


class SingleTaskMultivariateTPE(BaseTPE):
    def __init__(self,
                 hp_utils,
                 opt_requirements,
                 experimental_settings,
                 obj=None,
                 n_ei_candidates=24,
                 rule="scott",
                 gamma_func=default_gamma,
                 weight_func=default_weights):

        super().__init__(hp_utils,
                         opt_requirements,
                         experimental_settings,
                         obj=obj,
                         n_ei_candidates=n_ei_candidates,
                         rule=rule,
                         gamma_func=gamma_func,
                         weight_func=weight_func)

        self.opt = self.sample
        self.n_dimension = self.hp_utils.n_dimension

    def sample(self):
        hps_conf, _ = self.hp_utils.load_hps_conf(convert=True, do_sort=True, index_from_conf=False)
        hp_confs = []  # ndarray (D, n_ei_candidates)
        choices_list = []
        n_evals = len(hps_conf[0])
        n_lower = self.gamma_func(n_evals)
        basis_loglikelihoods_lower = np.zeros((self.n_dimension, self.n_ei_candidates, n_lower + 1))
        basis_loglikelihoods_upper = np.zeros((self.n_dimension, self.n_ei_candidates, n_evals - n_lower + 1))

        for dim, (var_name, hps) in enumerate(zip(self.hp_utils.var_names, hps_conf)):
            lower_hps, upper_hps = hps[:n_lower], hps[n_lower:]
            var_type = self.hp_utils.dist_types[var_name]

            if var_type in [float, int]:
                pe_lower, pe_upper = self._construct_numerical_parzen_estimator(var_name, var_type, lower_hps, upper_hps)
                choices_list.append(None)
            else:
                pe_lower, pe_upper, choices = self._construct_categorical_parzen_estimator(var_name, lower_hps, upper_hps)
                choices_list.append(choices)
            samples_lower = pe_lower.sample_from_density_estimator(self.rng, self.n_ei_candidates)
            hp_confs.append(samples_lower)
            basis_loglikelihoods_lower[dim] += pe_lower.basis_loglikelihood(samples_lower)
            basis_loglikelihoods_upper[dim] += pe_upper.basis_loglikelihood(samples_lower)

        hp_conf = self._compare_configurations(basis_loglikelihoods_lower, basis_loglikelihoods_upper, hp_confs, choices_list)

        return self.hp_utils.revert_hp_conf(hp_conf)

    def _calculate_conf_loglikelihood(self, basis_loglikelihood):
        """
        calculate loglikelihood of multivariate parzen estimator

        Parameters
        ----------
        basis_loglikelihood: ndarray (D=n_dim, N=n_ei_candidates, B=n_basis)
            Each element is the loglikelihood of D-th dimension's hyperparameter's
            parzen estimator's B-th basis of sample N.

        Returns
        -------
        loglikelihood of each configuration: ndarray (n_ei_candidates, )
        """

        conf_basis_loglikelihood = basis_loglikelihood.sum(axis=0)  # -> (N, B)
        (n_confs, n_basis) = conf_basis_loglikelihood.shape
        weights = self.weight_func(n_basis)
        weights /= weights.sum()
        ll_confs = logsumexp(conf_basis_loglikelihood, b=weights, axis=1)  # -> (N, )

        return ll_confs

    def _compare_configurations(self, basis_loglikelihood_lower, basis_loglikelihood_upper, hp_confs, choices_list):
        ll_lower = self._calculate_conf_loglikelihood(basis_loglikelihood_lower)
        ll_upper = self._calculate_conf_loglikelihood(basis_loglikelihood_upper)
        hp_confs = np.array(hp_confs)

        best_idx = np.argmax(ll_lower - ll_upper)
        raw_best_conf = hp_confs[:, best_idx]
        hp_conf = [hp_value if choices is None
                   else choices[int(hp_value)] for choices, hp_value in zip(choices_list, raw_best_conf)]

        return hp_conf
