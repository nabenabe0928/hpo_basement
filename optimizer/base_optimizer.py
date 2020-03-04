import numpy as np
import utils
import os
import csv
import time
import datetime
import sys
import subprocess as sp
import obj_functions.machine_learning_utils as ml_utils
from multiprocessing import Process
from typing import NamedTuple


"""
Parameters
----------
n_parallels: int
    the number of computer resources we use in an experiment
n_init: int
    the number of initial configurations
n_experiments: int
    the index of experiments. Used only to specify the path of log file.
is_barrier: bool
    if using the barrier function or not.
    if True, set inf or -inf when the parameter is infeasible.
    else, force the parameter set into the feasible domain.
restart: bool
    if restart the experiment or not.
    if True, continue the experiment based on log files.
default: bool
    If using default hyperparameter configuration or not.
max_evals: int
    the number of evlauations in an experiment.
seed: int or None
    The number specifying the seed on a random number generator.
verbose: bool
    Whether print the result or not.
print_freq: int
    Every print_freq iteration, the result will be printed.
check: bool
    If asking when removing files or not at the initialization.
cuda: list of int
    Which CUDA devices you use in the experiment. (Specify the single or multiple number(s))
transfer_info_paths: list of str
    The list of the path of previous information to transfer. 'opt/function/number'
"""


class BaseOptimizerRequirements(
    NamedTuple("_BaseOptimizerRequirements",
               [("n_parallels", int),  # 1
                ("n_init", int),  # 10
                ("n_experiments", int),  # 0
                ("max_evals", int),  # 100
                ("is_barrier", bool),  # True
                ("restart", bool),  # True
                ("default", bool),  # False
                ("seed", int),  # None
                ("verbose", bool),  # True
                ("print_freq", int),  # 1
                ("check", bool),  # False
                ("cuda", list),  # [0]
                ("transfer_info_paths", list),  # []
                ])):
    pass


def objective_function(hp_conf, hp_utils, cuda_id, job_id, is_barrier=True, verbose=True, print_freq=1, save_time=None):
    """
    Parameters
    ----------
    hp_conf: dict
        a hyperparameter configuration
    cuda_id: int
        the index of gpu used in an evaluation
    """

    save_path = "history/stdo" + hp_utils.save_path[11:] + "/log{:0>5}.csv".format(job_id)

    if is_barrier:
        is_out_of_domain = hp_utils.out_of_domain(hp_conf)
    else:
        is_out_of_domain = False
        hp_conf = hp_utils.pack_into_domain(hp_conf)

    eval_start = time.time()

    if hp_utils.in_fmt == "dict":
        hp_conf = hp_utils.list_to_dict(hp_conf)
        ml_utils.print_config(hp_conf, save_path, is_out_of_domain=is_out_of_domain)
    else:
        pass

    ys = {yn: yu for yn, yu in zip(hp_utils.y_names, hp_utils.y_upper_bounds)} \
        if is_out_of_domain else hp_utils.obj_class(hp_conf, cuda_id, save_path)

    hp_utils.save_hp_conf(hp_conf, ys, job_id)

    save_time(eval_start, hp_utils.lock, job_id)
    record_login(hp_utils.save_path)

    if verbose and job_id % print_freq == 0:
        utils.print_result(hp_conf, ys, job_id, hp_utils.list_to_dict)


def add_transfer_information(obj_path_name, transfer_info_paths):
    obj_path_name += "_transfers"
    n_tasks = len(transfer_info_paths)
    for i, path in enumerate(transfer_info_paths):
        # p = "{}{}_{}".format(*path[12:].split("/"))
        p = "{}_on_{}".format(*path[12:].split("/")[:-1])
        obj_path_name += "_" + p
        obj_path_name += "_and" if i + 1 < n_tasks else ""

    return obj_path_name


def record_login(save_path):
    current_time = str(datetime.datetime.today())[:-7]
    current_pid = os.environ["JOB_ID"] if "JOB_ID" in os.environ.keys() else os.getpid()
    record_path = "{0}/stdo/{2}/{3}/{4}/last_login.csv".format(*save_path.split("/"))

    with open(record_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["{}+{}".format(current_pid, current_time)])


def get_path_name(obj_name, experimental_settings, transfer_info_paths):
    obj_path_name = obj_name

    if experimental_settings.dim is not None:
        obj_path_name += "_{}d".format(experimental_settings.dim)
    if experimental_settings.dataset_name is not None:
        obj_path_name += "_{}".format(experimental_settings.dataset_name)
    if experimental_settings.n_cls is not None:
        obj_path_name += str(experimental_settings.n_cls)
    if experimental_settings.image_size is not None:
        obj_path_name += "_img{}".format(experimental_settings.image_size)
    if experimental_settings.data_frac is not None:
        obj_path_name += "_{}per".format(int(100 * experimental_settings.data_frac))
    if experimental_settings.biased_cls is not None:
        obj_path_name += "_biased"
    if experimental_settings.test:
        obj_path_name += "_TEST"
    if experimental_settings.all_train:
        obj_path_name += "_TrainAll"
    if transfer_info_paths is not None:
        obj_path_name = add_transfer_information(obj_path_name, transfer_info_paths)
    if experimental_settings.extra_exp_name is not None:
        obj_path_name += "_{}".format(experimental_settings.extra_exp_name)

    return obj_path_name


class BaseOptimizer():
    def __init__(self,
                 hp_utils,
                 requirements,
                 experimental_settings,
                 obj=None,
                 given_default=None):
        """
        Member Variables
        ----------------
        hp_utils: HyperparametersUtilities object
            ./utils/hp_utils.py/HyperparameterUtilities
        obj: function
            the objective function whose input is a hyperparameter configuration
            and output is the corresponding performance.
        save_file_path: string
            the path where recording the configurations and performances
        n_jobs: int
            the number of evaluations up to now
        opt: function
            the optimizer of hyperparameter configurations
        rng: numpy.random.RandomState object
            Sampling random numbers based on the seed argument.
        transfer_info_paths: list of str (M - 1, )
            The list of paths where there are previous log you want to transfer.
        """

        self.hp_utils = hp_utils
        self.obj = obj if obj is not None else objective_function
        self.max_evals = requirements.max_evals
        self.n_init = requirements.n_init
        self.n_parallels = max(requirements.n_parallels, 1)
        self.opt = callable
        self.restart = requirements.restart
        self.default = requirements.default
        self.given_default = self.format_default_confs(given_default)
        self.check = requirements.check
        self.rng = np.random.RandomState(requirements.seed)
        self.seed = requirements.seed
        self.cuda_id = requirements.cuda
        self.is_barrier = requirements.is_barrier
        opt_name = self.__class__.__name__ if not self.default else "DefaultConfs"
        opt_name += "" if experimental_settings.extra_opt_name is None else "_{}".format(experimental_settings.extra_opt_name)
        obj_path_name = get_path_name(self.hp_utils.obj_name, experimental_settings, requirements.transfer_info_paths)
        self.hp_utils.save_path = "history/log/{}/{}/{:0>3}".format(opt_name, obj_path_name, requirements.n_experiments)
        self.n_jobs = 0
        self.verbose = requirements.verbose
        self.print_freq = requirements.print_freq

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

        hps = self.hp_utils.config_space._hyperparameters
        sample = [None for _ in range(len(hps))]

        for var_name, hp in hps.items():
            idx = self.hp_utils.var_names.index(var_name)
            dist = self.hp_utils.dist_types[var_name]

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

    def print_optimized_result(self, best_hp_conf, best_performance):
        print("#### Best Configuration ####")
        print(best_hp_conf)
        print("##### Best Performance #####")
        print("{:.3f}".format(best_performance))

    def order_default_conf(self, conf):
        name_to_idx = self.hp_utils.config_space._hyperparameter_idx
        n_dim = len(name_to_idx)

        if len(conf) != n_dim:
            raise ValueError("The length of Default configurations must be eqaul to {}, but {} was given.".format(n_dim, len(conf)))
        elif eval(self.hp_utils.in_fmt) is not type(conf):
            raise TypeError("Default configurations must be identical to in_fmt specified in params.json.")

        if self.hp_utils.in_fmt == "list":
            return conf
        else:
            return_conf = [None for _ in range(len(name_to_idx))]
            for var_name, idx in name_to_idx.items():
                return_conf[idx] = conf[var_name]
            return return_conf

    def format_default_confs(self, given_conf):
        if given_conf is None:
            return None
        elif not type(given_conf) in [dict, list]:
            raise ValueError("Default configurations must be list or dict, but {} was given.".format(type(given_conf)))
        elif type(given_conf) == list and not type(given_conf[0]) in [dict, list]:
            return self.order_default_conf(given_conf)
        elif type(given_conf) == dict and not type(given_conf[list(given_conf.keys())[0]]) in [dict, list]:
            return self.order_default_conf(given_conf)
        elif type(given_conf) == list:
            return_confs = []
            for conf in given_conf:
                return_confs.append(self.order_default_conf(conf))
            return return_confs
        elif type(given_conf) == dict:
            return_confs = []
            n_conf = len(given_conf[list(given_conf.keys())[0]])
            reshaped_conf = [{k: v[i] for k, v in given_conf.items()} for i in range(n_conf)]
            for conf in reshaped_conf:
                return_confs.append(self.order_default_conf(conf))
            return return_confs

    def get_from_given_default(self):
        if self.given_default is None:
            return []
        elif type(self.given_default[0]) is list:
            n_given_confs = len(self.given_default)
            return self.given_default[self.n_jobs % n_given_confs]
        elif type(self.given_default) is list:
            return self.given_default
        else:
            raise ValueError("Given default configurations must be 1d or 2d array.")

    def check_double_submission(self):
        is_abci = "JOB_ID" in os.environ.keys()
        current_time = str(datetime.datetime.today())[:-7]
        save_path = "{0}/stdo/{2}/{3}/{4}/last_login.csv".format(*self.hp_utils.save_path.split("/"))
        job_ids = [] if is_abci \
            else [s.strip() for s in sp.check_output('ps -u {} -o pid'.format(os.environ["USER"]), shell=True).decode("utf-8").split("\n")
                  if s.strip().isdecimal()]
        # sp.check_output('qstat | cut -d " " -f 4 | sort -n', shell=True).split("\n") if is_abci \

        if os.path.isfile(save_path):
            with open(save_path, "r", newline="") as f:
                reader = list(csv.reader(f, delimiter="+"))[-1]
                last_pid, last_time = reader[0], reader[1]

            last_time_dt = datetime.datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")
            current_time_dt = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
            time_diff = current_time_dt - last_time_dt

            try:
                date_diff = time_diff.date
            except AttributeError:
                date_diff = 0
            try:
                second_diff = time_diff.seconds
            except AttributeError:
                second_diff = 0
        else:
            date_diff, second_diff, last_pid = 1, 1e+8, None

        if (not is_abci and last_pid not in job_ids) or (is_abci and date_diff > 0 or second_diff > self.hp_utils.waiting_time):
            record_login(self.hp_utils.save_path)
        else:
            print("You are running the same program in different processes.\nWill Stop this process to prevent double evalutions.")
            sys.exit()

    def optimize(self):
        utils.create_log_dir(self.hp_utils.save_path)
        if not self.restart:
            utils.check_conflict(self.hp_utils.save_path, check=self.check)
        utils.create_log_dir(self.hp_utils.save_path)
        dy = {y_name: None for y_name in self.hp_utils.y_names}
        self.hp_utils.save_hp_conf(list(range(len(self.hp_utils.var_names))), dy, None, record=False)

        self.n_jobs = self.get_n_jobs()
        self.check_double_submission()

        save_time = utils.save_elapsed_time(self.hp_utils.save_path, self.hp_utils.lock, self.verbose, self.print_freq)

        if self.n_parallels <= 1:
            self._optimize_sequential(save_time)
        else:
            self._optimize_parallel(save_time)

        hps_conf, losses = self.hp_utils.load_hps_conf(do_sort=True)
        best_hp_conf, best_performance = hps_conf[0], losses[0][0]
        self.print_optimized_result(best_hp_conf, best_performance)

        return best_hp_conf, best_performance

    def _optimize_sequential(self, save_time):
        while True:
            if self.default and self.n_jobs < self.max_evals:
                hp_conf = self.get_from_given_default()
            elif self.n_jobs < self.n_init:
                hp_conf = self._initial_sampler()
            elif self.n_jobs < self.max_evals:
                hp_conf = self.opt()
            else:
                break

            self.obj(hp_conf,
                     self.hp_utils,
                     self.cuda_id[0],
                     self.n_jobs,
                     is_barrier=self.is_barrier,
                     verbose=self.verbose,
                     print_freq=self.print_freq,
                     save_time=save_time)

            self.n_jobs += 1
            time.sleep(1.0e-3)

    def _optimize_parallel(self, save_time):
        jobs = []
        n_runnings = 0
        resources = [False for _ in range(self.n_parallels)]

        while True:
            new_jobs = []

            for job in jobs:
                if job[1].is_alive():
                    new_jobs.append(job)
                    resources[job[0]] = True
                else:
                    resources[job[0]] = False

            jobs = new_jobs
            n_runnings = len(jobs)
            for _ in range(max(0, self.n_parallels - n_runnings)):
                cidx = resources.index(False)

                if self.default and self.n_jobs < self.max_evals:
                    hp_conf = self.get_from_given_default()
                elif self.n_jobs < self.n_init:
                    hp_conf = self._initial_sampler()
                else:
                    hp_conf = self.opt()

                p = Process(target=self.obj,
                            args=(hp_conf,
                                  self.hp_utils,
                                  self.cuda_id[cidx],
                                  self.n_jobs,
                                  self.is_barrier,
                                  self.verbose,
                                  self.print_freq,
                                  save_time))

                resources[cidx] = True
                p.start()
                jobs.append([cidx, p])
                self.n_jobs += 1

                if self.n_jobs >= self.max_evals:
                    return 0

            time.sleep(1.0e-3)
