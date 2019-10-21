import numpy as np
import utils
from optimizer.parzen_estimator import NumericalParzenEstimator, CategoricalParzenEstimator
from optimizer import BaseOptimizer


EPS = 1e-12


def default_gamma(x, n_samples_lower=25):

    return min(int(np.ceil(0.25 * np.sqrt(x))), n_samples_lower)


def default_weights(x, n_samples_lower=25):
    if x == 0:
        return np.asarray([])
    elif x < n_samples_lower:
        return np.ones(x)
    else:
        ramp = np.linspace(1.0 / x, 1.0, num=x - n_samples_lower)
        flat = np.ones(n_samples_lower)
        return np.concatenate([ramp, flat], axis=0)


class SingleTaskTPE(BaseOptimizer):
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
                 n_ei_candidates=24,
                 gamma_func=default_gamma,
                 weight_func=default_weights):
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

        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weight_func = weight_func
        self.opt = self.sample

    def sample(self):
        hps_conf, _ = self.hp_utils.load_hps_conf(convert=True, do_sort=True, index_from_conf=False)
        n_lower = self.gamma_func(len(hps_conf))
        hp_conf = []

        for idx, hps in enumerate(hps_conf):
            lower_vals, upper_vals = hps[:n_lower], hps[n_lower:]
            var_name = self.hp_utils.config_space._idx_to_hyperparameter[idx]
            var_type = utils.distribution_type(self.hp_utils.config_space, var_name)

            if var_type in [str, bool]:
                cat_idx = self._sample_categorical(lower_vals, upper_vals)
                hp_value = self.hp.choices[cat_idx]
            elif var_type in [float, int]:
                hp_value = self._sample_numerical(var_name, var_type, lower_vals, upper_vals)
            hp_conf.append(hp_value)

        return self.hp_utils.revert_hp_conf(hp_conf)

    def _sample_numerical(self, var_name, var_type, lower_vals, upper_vals):
        hp = self.hp_utils.config_space._hyperparameters[var_name]
        q, log = hp.q, hp.log
        lb, ub = 0., 1.
        converted_q = None

        if var_type is int or q is not None:
            if not log:
                converted_q = 1. / (hp.upper - hp.lower) if q is None else q / (hp.upper - hp.lower)
                lb -= 0.5 * converted_q
                ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(lower_vals, lb, ub, self.weight_func, q=converted_q)
        pe_upper = NumericalParzenEstimator(upper_vals, lb, ub, self.weight_func, q=converted_q)

        return var_type(self._compare_candidates(pe_lower, pe_upper))

    def _sample_categorical(self, var_name, lower_vals, upper_vals):
        choices = self.hp_utils._hyperparameters[var_name].choices
        n_choices = len(choices)
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        pe_lower = CategoricalParzenEstimator(lower_vals, n_choices, self.weight_func)
        pe_upper = CategoricalParzenEstimator(upper_vals, n_choices, self.weight_func)

        return int(self._compare_candidates(pe_lower, pe_upper))

    def _compare_candidates(self, pe_lower, pe_upper):
        samples_lower = pe_lower.sample_from_density_estimator(self.rng, self.n_ei_candidates)
        ll_lower = pe_lower.log_likelihood(samples_lower)
        ll_upper = pe_upper.log_likelihood(samples_lower)

        best_idx = np.argmax(ll_lower - ll_upper)

        return samples_lower[best_idx]
