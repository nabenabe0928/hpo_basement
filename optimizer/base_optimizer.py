import numpy as np
import utils
import os
import csv
import time
from multiprocessing import Process, Lock


def objective_function(hp_conf, hpu, n_gpu, n_jobs):
    """
    Parameters
    ----------
    hp_conf: dict
        a hyperparameter configuration
    n_gpu: int
        the index of gpu used in an evaluation
    """

    if utils.out_of_domain(hp_conf, hpu):
        hpu.save_hps(hp_conf, {yn: 1.0e+8 for yn in hpu.y_names}, n_jobs, hpu.lock)
    else:
        ys = hpu.obj_class(hp_conf, n_gpu)
        hpu.save_hps(hp_conf, ys, n_jobs)


class BaseOptimizer():
    """
    Parameters
    ----------
    hpu: HyperparametersUtilities object
        ./utils/hp_utils.py/HyperparameterUtilities
    rs: bool
        if random search or not
    obj: function
        the objective function whose input is a hyperparameter configuration
        and output is the corresponding performance.
    n_parallels: int
        the number of computer resources we use in an experiment
    n_init: int
        the number of initial configurations
    max_evals: int
        the number of evlauations in an experiment
    """

    def __init__(self, hpu, rs=True, n_parallels=1, n_init=10, max_evals=100, obj=objective_function):
        """
        Member Variables
        ----------------
        save_file_path: string
            the path where recording the configurations and performances
        n_jobs: int
            the number of evaluations up to now
        opt: function
            the optimizer of hyperparameter configurations
        """

        self.hpu = hpu
        self.n_jobs = self.get_n_jobs()
        self.obj = obj
        self.n_parallels = hpu.n_parallels
        self.cs = hpu.config_space
        self.save_file_path = hpu.save_path
        self.max_evals = max_evals
        self.n_init = n_init
        self.n_parallels = n_parallels
        self.opt = callable
        self.rs = rs
        
    def get_n_jobs(self):
        """
        The function to get currenct number of evaluations in the beginning of restarting of an experiment 
        
        Returns
        -------
        The number of evaluations
        """

        param_files = os.path.isdir(self.save_file_path)
        n_jobs = 0
        if len(param_files) > 0:
            for param_file in param_files:
                with open(self.save_file_path + param_file, "r", newline="") as f:
                    n_jobs = max(len(list(csv.reader(f, delimiter=","))))
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
            dist = utils.distribution_type(self.cs, var_name)
            if dist == str:
                # categorical
                choices = hp.choices
                rnd = np.random.randint(len(choices))
                sample[idx] = choices[rnd]
            else:
                # numerical
                rnd = np.random.random()
                sample[idx] = utils.revert_hp(rnd, self.cs, var_name)

        return self.hpu.list_to_dict(sample)

    def optimize(self):
        if self.n_parallels <= 1:
            self._optimize_sequential()
        else:
            self._optimize_parallel()

    def _optimize_sequential(self):
        while True:
            n_gpu = 0

            if self.n_jobs < self.n_init or self.rs:
                hp_conf = self._initial_sampler()
            else:
                hp_conf = self.opt()
            self.obj(hp_conf, self.hpu, n_gpu, self.n_jobs)
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
                n_gpu = gpus.index(False)

                if self.n_jobs < self.n_init or self.rs:
                    hp_conf = self._initial_sampler()
                else:
                    hp_conf = self.opt()

                p = Process(target=self.obj, args=(hp_conf, self.hpu, n_gpu, self.n_jobs))
                p.start()
                jobs.append([n_gpu, p])
                self.n_jobs += 1

                if self.n_jobs >= self.max_evals:
                    break

                time.sleep(1.0e-4)

            if self.n_jobs >= self.max_evals:
                break
