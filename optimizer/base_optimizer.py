import numpy as np
import utils
import os
import csv
import time
from multiprocessing import Process


def objective_function(hp_conf, hp_utils, gpu_id, job_id):
    """
    Parameters
    ----------
    hp_conf: dict
        a hyperparameter configuration
    gpu_id: int
        the index of gpu used in an evaluation
    """

    if hp_utils.out_of_domain(hp_conf):
        hp_utils.save_hp_conf(hp_conf, {yn: 1.0e+8 for yn in hp_utils.y_names}, job_id)
    else:
        ys = hp_utils.obj_class(hp_conf, gpu_id)
        hp_utils.save_hp_conf(hp_conf, ys, job_id)


class BaseOptimizer():
    """
    Parameters
    ----------
    hp_utils: HyperparametersUtilities object
        ./utils/hp_utils.py/HyperparameterUtilities
    obj: function
        the objective function whose input is a hyperparameter configuration
        and output is the corresponding performance.
    n_parallels: int
        the number of computer resources we use in an experiment
    n_init: int
        the number of initial configurations
    n_experiments: int
        the index of experiments. Used only to specify the path of log file.
    restart: bool
        if restart the experiment or not.
        if True, continue the experiment based on log files.
    max_evals: int
        the number of evlauations in an experiment
    seed: int or None
        The number specifying the seed on a random number generator
    """

    def __init__(self,
                 hp_utils,
                 n_parallels=1,
                 n_init=10,
                 n_experiments=0,
                 max_evals=100,
                 restart=True,
                 seed=None,
                 obj=objective_function):
        """
        Member Variables
        ----------------
        save_file_path: string
            the path where recording the configurations and performances
        n_jobs: int
            the number of evaluations up to now
        opt: function
            the optimizer of hyperparameter configurations
        rng: numpy.random.RandomState object
            Sampling random numbers based on the seed argument.
        ongoing_confs: 2d list [gpu_id][hp idx]
            Hyperparameter configurations being evaluated now.
        """

        self.hp_utils = hp_utils
        self.obj = obj
        self.cs = hp_utils.config_space
        self.max_evals = max_evals
        self.n_init = n_init
        self.n_parallels = max(n_parallels, 1)
        self.opt = callable
        self.restart = restart
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.ongoing_confs = [None for _ in range(self.n_parallels)]
        opt_name = self.__class__.__name__
        self.hp_utils.save_path = "history/log/{}/{}/{:0>3}".format(opt_name, hp_utils.obj_name, n_experiments)
        self.n_jobs = 0

    def get_n_jobs(self):
        """
        The function to get currenct number of evaluations in the beginning of restarting of an experiment

        Returns
        -------
        The number of evaluations at the beginning of restarting of an experiment.
        """

        param_files = os.listdir(self.hp_utils.save_path)
        n_jobs = 0
        if len(param_files) > 0:
            with open(self.hp_utils.save_path + "/" + self.hp_utils.y_names[0] + ".csv", "r", newline="") as f:
                n_jobs = len(list(csv.reader(f, delimiter=",")))
        else:
            n_jobs = 0

        return n_jobs

    def _initial_sampler(self):
        """
        random sampling for an initialization

        Returns
        -------
        hyperparameter configurations: list
        """

        hps = self.cs._hyperparameters
        sample = [None for _ in range(len(hps))]

        for var_name, hp in hps.items():
            idx = self.cs._hyperparameter_idx[var_name]
            dist = self.hp_utils.distribution_type(var_name)
            if dist is str or dist is bool:
                # categorical
                choices = hp.choices
                rnd = self.rng.randint(len(choices))
                sample[idx] = choices[rnd]
            else:
                # numerical
                rnd = self.rng.uniform()
                sample[idx] = self.hp_utils.revert_hp(rnd, var_name)

        return sample

    def optimize(self):
        utils.create_log_dir(self.hp_utils.save_path)
        if not self.restart:
            utils.check_conflict(self.hp_utils.save_path)
        utils.create_log_dir(self.hp_utils.save_path)

        self.n_jobs = self.get_n_jobs()

        if self.n_parallels <= 1:
            self._optimize_sequential()
        else:
            self._optimize_parallel()

    def _optimize_sequential(self):
        while True:
            gpu_id = 0

            if self.n_jobs < self.n_init:
                hp_conf = self._initial_sampler()
            else:
                hp_conf = self.opt()

            self.ongoing_confs[0] = hp_conf[:]
            self.obj(hp_conf, self.hp_utils, gpu_id, self.n_jobs)
            self.n_jobs += 1

            if self.n_jobs >= self.max_evals:
                break

    def _optimize_parallel(self):
        jobs = []
        n_runnings = 0

        while True:
            gpus = [False for _ in range(self.n_parallels)]
            if len(jobs) > 0:
                n_runnings = 0
                new_jobs = []
                for job in jobs:
                    if job[1].is_alive():
                        new_jobs.append(job)
                        gpus[job[0]] = True
                jobs = new_jobs
                n_runnings = len(jobs)
            else:
                n_runnings = 0

            for _ in range(max(0, self.n_parallels - n_runnings)):
                gpu_id = gpus.index(False)
                self.ongoing_confs[gpu_id] = None

                if self.n_jobs < self.n_init:
                    hp_conf = self._initial_sampler()
                else:
                    hp_conf = self.opt()

                self.ongoing_confs[gpu_id] = hp_conf[:]
                p = Process(target=self.obj, args=(hp_conf, self.hp_utils, gpu_id, self.n_jobs))
                p.start()
                jobs.append([gpu_id, p])
                self.n_jobs += 1

                if self.n_jobs >= self.max_evals:
                    break

                time.sleep(1.0e-4)

            if self.n_jobs >= self.max_evals:
                break
