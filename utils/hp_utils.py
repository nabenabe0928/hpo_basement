import numpy as np
import json
import csv
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import utils
from multiprocessing import Lock


def create_hyperparameter(var_type, name, lower=None, upper=None, log=False, q=None, choices=None):
    """
    Parameters
    ----------
    var_type: type
        int or float or str
    dist: string
        "u" (uniform) or "c" (categorical)
    q: float or int
        any positive real number
    lower: float or int
        upper bound of the variable
    upper: float or int
        lower bound of the variable
    choices: list of str or float or int
        the choices of categorical parameter

    Returns
    -------
    ConfigSpace.hyperparameters object
        the information of hyperparameter
    """

    if var_type == int:
        return CSH.UniformIntegerHyperparameter(name=name, lower=lower, upper=upper, log=log, q=q)
    elif var_type == float:
        return CSH.UniformFloatHyperparameter(name=name, lower=lower, upper=upper, log=log, q=q)
    elif var_type == str or var_type == bool:
        return CSH.CategoricalHyperparameter(name=name, choices=choices)
    else:
        raise ValueError("The hp_type must be chosen from [int, float, str, bool]")


def get_hp_info(hp):
    """
    Parameters
    ----------
    hp: ConfigSpace.hyperparameters object
        the information of a hyperparameter

    Returns
    -------
    lower: float or int
        the lower bound of a hyperparameter in the searching space
    upper: float or int
        the upper bound of a hyperparameter in the searching space
    """

    try:
        if not hp.log:
            return hp.lower, hp.upper, hp.q, hp.log
        else:
            return np.log(hp.lower), np.log(hp.upper), hp.q, hp.log
    except NotImplementedError:
        raise NotImplementedError("Categorical parameters do not have the log scale option.")


def save_hp(save_file_path, lock, job_id, value):
    """
    recording a hyperparameter evaluated in an experiment

    Parameters
    ----------
    save_file_path: string
        the path of a file to record a hyperparameter
    lock: multiprocessing object
        preventing multiple accesses to a file
    job_id: int
        the number of evaluations in an experiment
    value: float or int or string
        the value of a hyperparameter evaluated in this iteration
    """

    if not os.path.isfile(save_file_path):
        with open(save_file_path, "w", newline="") as f:
            pass

    lock.acquire()
    with open(save_file_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([job_id, value])
    lock.release()


def load_hps(save_file_path, lock, var_type):
    """
    loading hyperparameters evaluated in an experiment

    Parameters
    ----------
    var_type: type
        the type of hyperparameter

    Returns
    -------
    the list of a hyperparameter evlauated in an experiment
    """

    lock.acquire()
    with open(save_file_path, "r", newline="") as f:
        reader = [var_type(row[1]) for row in list(csv.reader(f, delimiter=","))]
    lock.release()
    return reader


class HyperparameterUtilities():
    """
    Parameters
    ----------
    obj_name: string
        The name of the objective function's file
    dim: int
        the dimension of a benchmark function. Required only when using a benchmark function.
    """

    def __init__(self, obj_name, dim=None):
        """
        Member Variables
        config_space: ConfigurationSpace
            config space that contains the hyperparameter information
        obj_class: class
            The class of the objective function
        y_names: list of string
            the names of the measurements of hyperparameter configurations
        save_path: string
            the path where we save hyperparameter configurations and corresponding performances.
            history/{log, stdo}/name of optimizer/name of algorithm/number of experiments
        """

        self.obj_name = obj_name
        self.config_space = CS.ConfigurationSpace()
        self.y_names = None
        self.lock = Lock()
        self.dim = dim
        self.save_path = None
        self.obj_class = self.prepare_opt_env()
        self.var_names = list(self.config_space._hyperparameters.keys())

    def dict_to_list(self, hp_dict):
        """
        converting a dict of a hyperparameter configuration into a list

        Parameters
        ----------
        hp_dict: list of a hyperparameter configuration
            indexes follow the config space and values are corresponding values.

        Returns
        -------
        list of a hyperparameter configuration
            the indexes follow the config space
        """

        if type(hp_dict) != dict:
            raise NotImplementedError("{} is not dict.".format(hp_dict))

        hp_list = [None for _ in range(len(hp_dict))]

        for var_name, value in hp_dict.items():
            idx = self.config_space._hyperparameter_idx[var_name]
            hp_list[idx] = value

        if None in hp_list:
            raise ValueError("hp_list is including None.")

        return hp_list

    def list_to_dict(self, hp_list):
        """
        converting a list of a hyperparameter configuration into a dict

        Parameters
        ----------
        hp_list: list of a hyperparameter configuration
            indexes follow the config space and values are corresponding values

        Returns
        -------
        dict of a hyperparameter configuration
            the keys follow the config space
        """

        if type(hp_list) != list:
            raise NotImplementedError("{} is not list.".format(hp_list))

        hp_dict = {}

        for idx, value in enumerate(hp_list):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            hp_dict[var_name] = value

        return hp_dict

    def distribution_type(self, var_name):
        """
        Parameters
        ----------
        var_name: str
            the name of target hyperparameter.

        Returns
        -------
        type of hyperparameter
        """

        cs_dist = str(type(self.config_space._hyperparameters[var_name]))

        if "Integer" in cs_dist:
            return int
        elif "Float" in cs_dist:
            return float
        elif "Categorical" in cs_dist:
            var_type = type(self.config_space._hyperparameters[var_name].choices[0])
            if var_type == str or var_type == bool:
                return var_type
            else:
                raise ValueError("The type of categorical parameters must be 'bool' or 'str'.")
        else:
            raise NotImplementedError("The distribution is not implemented.")

    def out_of_domain(self, hp_conf):
        """
        Parameters
        ----------
        hp_conf: list or dict of hyperparameter values
            if list, the indexes follows the indexes in the ConfigSpace
            else if dict, the keys are the name of hyperparameters
            values are the value of hyperparameter which will be evaluated.

        Returns
        -------
        boolean
            if the value of hyperparameter is out of domain, True.
            otherwise, False.
        """

        hp_dict = self.list_to_dict(hp_conf) if type(hp_conf) == list else hp_conf

        for var_name, value in hp_dict.items():
            hp = self.config_space._hyperparameters[var_name]

            try:
                lb, ub = hp.lower, hp.upper
                if lb <= value <= ub:
                    pass
                else:
                    return True
            except ValueError:
                # categorical parameters
                pass
        return False

    def convert_hp(self, hp_value, var_name):
        """
        converting the value of hyperparameter in [0, 1]

        Parameters
        ----------
        hp_value: int or float
            the value of a hyperparameter
        var_name: string
            the name of a hyperparameter

        Returns
        -------
        float or int value
            the value is constrained in [0, 1]
        """

        try:
            lb, ub, _, log = get_hp_info(self.config_space._hyperparameters[var_name])
            hp_value = np.log(hp_value) if log else hp_value
            return (hp_value - lb) / (ub - lb)
        except NotImplementedError:
            raise NotImplementedError("Categorical parameters do not have lower and upper options.")

    def convert_hp_conf(self, hp_conf):
        """
        Converting each value of a hyperparameter configuration into [0, 1]

        Parameters
        ----------
        hp_conf: list
            one hyperparameter configuration

        Returns
        -------
        1d list of float or int value
            the values are constrained in [0, 1]
        """

        hp_converted_conf = []
        for idx, hp_value in enumerate(hp_conf):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            d = self.distribution_type(var_name)
            if d is int or d is float:
                hp_converted_conf.append(self.convert_hp(hp_value, var_name))
            else:
                hp_converted_conf.append(hp_value)
        return hp_converted_conf

    def convert_hp_confs(self, hp_confs):
        """
        Converting each value of hyperparameter configurations into [0, 1]

        Parameters
        ----------
        hp_confs: list
            the list of hyperparameter configurations

        Returns
        -------
        2d list of float or int value
            the values are constrained in [0, 1]
        """

        hp_converted_confs = []
        for hp_conf in hp_confs:
            hp_converted_confs.append(self.convert_hp_conf(hp_conf))
        return hp_converted_confs

    def revert_hp(self, hp_converted_value, var_name):
        """
        reverting the value of hyperparameter into an original scale

        Parameters
        ----------
        hp_converted_value: int or float
            converted value of a hyperparameter
        var_name: string
            the name of a hyperparameter

        Returns
        -------
        float or int value
            the value in an original scale
        """

        try:
            lb, ub, q, log = get_hp_info(self.config_space._hyperparameters[var_name])
            var_type = self.distribution_type(var_name)
            hp_value = (ub - lb) * hp_converted_value + lb
            hp_value = np.exp(hp_value) if log else hp_value
            hp_value = np.floor(hp_value / q) * q if q is not None else hp_value
            return var_type(hp_value)
        except NotImplementedError:
            raise NotImplementedError("Categorical parameters do not have lower and upper options.")

    def revert_hp_conf(self, hp_converted_conf):
        """
        Reverting each value of a hyperparameter configuration into original scales

        Parameters
        ----------
        hp_converted_conf: list
            one hyperparameter configuration constrained in [0, 1]

        Returns
        -------
        1d list of float or int value
            the values of a hyperparameter configuration on original scales
        """

        hp_conf = []
        for idx, hp_converted_value in enumerate(hp_converted_conf):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            d = self.distribution_type(var_name)
            if d is int or d is float:
                hp_conf.append(self.revert_hp(hp_converted_value, var_name))
            else:
                hp_conf.append(hp_converted_value)

        return hp_conf

    def revert_hp_confs(self, converted_hp_confs):
        """
        Reverting each value of hyperparameter configurations into original scales

        Parameters
        ----------
        converted_hp_confs: list
            the list of hyperparameter configurations constrained in [0, 1]

        Returns
        -------
        2d list of float or int value
            the values of hyperparameter configurations in original scales
        """

        hp_confs = []
        for converted_hp_conf in converted_hp_confs:
            hp_confs.append(self.revert_hp_conf(converted_hp_conf))
        return hp_confs

    def save_hp_conf(self, hp_conf, ys, job_id, converted=False):
        """
        recording a hyperparameter configuration and the corresponding performance

        Parameters
        ----------
        hps: dict or list
            a hyperparameter configuration
        ys: dict
            the keys are the name of objective functions
            and the values are the corresponding values
            e.g.) {"loss" 0.67, "acc": 0.81}
        job_id: int
            the number of evaluations in an experiment
        converted: bool
            if True, reverting into original scales
        """

        if type(hp_conf) == dict:
            hp_conf = self.dict_to_list(hp_conf)
        if type(ys) != dict:
            raise ValueError("ys must be dict.")
        if converted:
            hp_conf = self.revert_hp_conf(hp_conf)
        for idx, hp in enumerate(hp_conf):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, hp)

        for var_name, v in ys.items():
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, v)

    def load_hps_conf(self, convert=False, do_sort=False):
        """
        loading hyperparameter configurations and the corresponding performance

        Parameters
        ----------
        convert: bool
            if True, converting into constrained scales

        Returns
        -------
        the list of hyperparameter configurations and the corresponding performance
        [the index for configurations][the index for hyperparameters]
        """

        cs = self.config_space
        names = cs._idx_to_hyperparameter
        hps_conf = []
        ys = []
        for idx in range(len(names)):
            var_name = names[idx]
            var_type = self.distribution_type(var_name)
            save_file_path = self.save_path + "/" + var_name + ".csv"
            hps = load_hps(save_file_path, self.lock, var_type)
            if convert:
                hps = [self.convert_hp(hp, var_name) for hp in hps]
            hps_conf.append(hps)

        for y_name in self.y_names:
            save_file_path = self.save_path + "/" + y_name + ".csv"
            y = load_hps(save_file_path, self.lock, float)
            ys.append(y)

        if do_sort:
            order = np.argsort(ys[0])
            ys = [np.array(y[order]) for y in ys]
            hps_conf = [np.array(hps[order]) for hps in hps_conf]
        n_confs = len(hps_conf[0])
        n_dims = len(hps_conf)
        hps_conf = [[hps_conf[nd][nc] for nd in range(n_dims)] for nc in range(n_confs)]

        return hps_conf, ys

    def prepare_opt_env(self):
        """
        The function to create ConfigSpace and load the objective function's class
        """

        with open("params.json") as f:
            json_params = json.load(f)[self.obj_name]

        config_info = json_params["config"]
        self.y_names = json_params["y_names"]

        if self.dim is not None:
            try:
                v = config_info['x']
            except ValueError:
                raise ValueError("ONLY 'x' is allowed to be the name of hyperparameters of benchmark functions in params.json.")
            lb, ub = v["lower"], v["upper"]
            for i in range(self.dim):
                var_name = 'x{:0>5}'.format(i)
                hp = create_hyperparameter(float, var_name, lower=lb, upper=ub)
                self.config_space.add_hyperparameter(hp)
        else:
            for var_name, v in config_info.items():
                hp = self.get_hp_info_from_json(v, var_name)
                self.config_space.add_hyperparameter(hp)

        fd, main_f = json_params["func_dir"], json_params["main"]

        return utils.load_class("obj_functions.{}.{}.{}".format(fd, self.obj_name, main_f))

    def get_hp_info_from_json(self, v, var_name):
        """
        Parameters
        ----------
        v: dict
            loaded from params.json
        var_name: str
            the name of a hyperparameter

        Returns
        -------
        hp: ConfigSpace.hyperparameters object
            The object including a hyperparameter's information
        """

        dist = v["dist"]
        vt = v["var_type"]

        if vt in ["int", "float", "str", "bool"]:
            vt = eval(vt)
        else:
            raise ValueError("var_type in params.json must be 'int' or 'float' or 'str' or 'bool'.")

        if dist == "u":
            if "q" not in v.keys() or v["q"] == "None":
                q = None
            elif type(v["q"]) == int or type(v["q"]) == float:
                q = v["q"]
            else:
                raise ValueError("q in params.json must be int or float or 'None'.")
            if "log" not in v.keys():
                log = False
            elif v["log"] == "True" or v["log"] == "False":
                log = eval(v["log"])
            else:
                raise ValueError("log in params.json must be 'True' or 'False'.")
            lb, ub = v["lower"], v["upper"]
            hp = create_hyperparameter(vt, var_name, lower=lb, upper=ub, log=log, q=q)
        elif dist == "c":
            choices = v["choices"]
            hp = create_hyperparameter(vt, var_name, choices=choices)
        else:
            raise ValueError("The first element of json hp dict must be 'u' (uniform) or 'c' (categorical).")

        return hp
