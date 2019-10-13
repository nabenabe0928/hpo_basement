import numpy as np
import json
import csv
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from utils.utils import load_class, create_log_dir
from multiprocessing import Lock


def create_hyperparameter(var_type, name, lower=None, upper=None, log=False, q=None, choices=None):
    """
    Parameters
    ----------
    var_type: type
        int or float or str
    dist: string
        "uniform" or "cat"
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
    elif var_type == str:
        return CSH.CategoricalHyperparameter(name=name, choices=choices)
    else:
        raise ValueError("The hp_type must be chosen from [int, float, cat]")


def distribution_type(cs, var_name):
    """
    Parameters
    ----------
    cs: ConfigSpace object
        Configspace containing the information of the hyperparameters
    var_name: str
        the name of target hyperparameter.

    Returns
    -------
    type of hyperparameter
    """

    cs_dist = str(type(cs._hyperparameters[var_name]))

    if "Integer" in cs_dist:
        return int
    elif "Float" in cs_dist:
        return float
    elif "Categorical" in cs_dist:
        return str
    else:
        raise NotImplementedError("The distribution is not implemented.")


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
        if hp.log:
            return hp.lower, hp.upper, hp.q, hp.log
        else:
            return np.log(hp.lower), np.log(hp.upper), hp.q, hp.log
    except:
        raise NotImplementedError("Categorical parameters do not have the log scale option.")


def out_of_domain(hps, hpu):
    """
    Parameters
    ----------
    hps: list or dict of hyperparameter values
        if list, the indexes follows the indexes in the ConfigSpace
        else if dict, the keys are the name of hyperparameters
        values are the value of hyperparameter which will be evaluated.  
    hpu: HyperparameterUtilities object

    Returns
    -------
    boolean
        if the value of hyperparameter is out of domain, True.
        otherwise, False.
    """

    hp_dict = hpu.list_to_dict(hps) if type(hps) == list else hps

    for var_name, value in hp_dict.items():
        hp = hpu.config_space._hyperparameters[var_name]
        
        try:
            l, u = hp.lower, hp.upper
            if l <= value <= u:
                pass
            else:
                return True
        except:
            # categorical parameters
            pass
    return False


def convert_hp(hp_value, cs, var_name):
    """
    converting the value of hyperparameter in [0, 1]

    Parameters
    ----------
    hp_value: int or float
        the value of a hyperparameter
    cs: ConfigSpace object
        the configuration space containing the information
    var_name: string
        the name of a hyperparameter

    Returns
    -------
    float or int value
        the value is constrained in [0, 1]
    """

    try:
        l, u, _, log = get_hp_info(cs._hyperparameters[var_name])
        hp_value = np.log(hp_value) if log else hp_value
        return (hp_value - l) / (u - l)
    except:
        raise NotImplementedError("Categorical parameters do not have lower and upper options.")


def convert_hps(hp_values, cs):
    """
    Converting each value of a hyperparameter configuration into [0, 1]

    Parameters
    ----------
    hp_values: list
        one hyperparameter configuration

    Returns
    -------
    list of float or int value
        the values are constrained in [0, 1]
    """

    hp_converted_values = []
    for idx, hp_value in enumerate(hp_values):
        var_name = cs._idx_to_hyperparameter[idx]
        if distribution_type(cs, var_name) is not str:
            hp_converted_values.append(convert_hp(hp_value, cs, var_name))
        else:
            hp_converted_values.append(hp_value)
    return hp_converted_values


def convert_hps_set(hps_set, cs):
    """
    Converting each value of hyperparameter configurations into [0, 1]

    Parameters
    ----------
    hps_set: list
        the list of hyperparameter configurations

    Returns
    -------
    list of float or int value
        the values are constrained in [0, 1]
    """

    hp_converted_values_set = []
    for hps in hps_set:
        hp_converted_values_set.append(convert_hps(hps, cs))
    return hp_converted_values_set


def revert_hp(hp_converted_value, cs, var_name):
    """
    reverting the value of hyperparameter into an original scale

    Parameters
    ----------
    hp_converted: int or float
        converted value of a hyperparameter
    cs: ConfigSpace object
        the configuration space containing the information
    var_name: string
        the name of a hyperparameter

    Returns
    -------
    float or int value
        the value in an original scale
    """

    try:
        l, u, q, log = get_hp_info(cs._hyperparameters[var_name])
        var_type = distribution_type(cs, var_name)
        hp_value = (u - l) * hp_converted_value + l
        hp_value = np.exp(hp_value) if log else hp_value
        hp_value = np.floor(hp_value / q) * q if q is not None else hp_value
        return var_type(hp_value)
    except:
        raise NotImplementedError("Categorical parameters do not have lower and upper options.")


def revert_hps(hp_converted_values, cs):
    """
    Reverting each value of a hyperparameter configuration into original scales

    Parameters
    ----------
    hp_converted_values: list
        one hyperparameter configuration constrained in [0, 1]

    Returns
    -------
    list of float or int value
        the values of a hyperparameter configuration on original scales
    """

    hp_values = []
    for idx, hp_converted_value in enumerate(hp_converted_values):
        var_name = cs._idx_to_hyperparameter[idx]
        if distribution_type(cs, var_name) is not str:
            hp_values.append(revert_hp(hp_converted_value, cs, var_name))
        else:
            hp_values.append(hp_converted_value)

    return hp_values


def revert_hps_set(converted_hps_set, cs):
    """
    Reverting each value of hyperparameter configurations into original scales

    Parameters
    ----------
    converted_hps_set: list
        the list of hyperparameter configurations constrained in [0, 1]

    Returns
    -------
    list of float or int value
        the values of hyperparameter configurations in original scales
    """

    hp_values_set = []
    for converted_hps in converted_hps_set:
        hp_values_set.append(revert_hps(converted_hps, cs))
    return hp_values_set


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


def load_hp(save_file_path, lock, var_type):
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
    opt_name: string
        the name of an optimizer
    y_names: list of string
        the names of the measurements of hyperparameter configurations
    """
    def __init__(self, obj_name, opt_name, n_experiments, y_names):
        """
        Member Variables
        config_space: ConfigurationSpace
            config space that contains the hyperparameter information
        obj_class: class
            The class of the objective function
        save_path: string
            the path where we save hyperparameter configurations and corresponding performances.
            history/{log, stdo}/name of optimizer/name of algorithm/number of experiments
        """

        self.obj_name = obj_name
        self.y_names = y_names
        self.config_space = CS.ConfigurationSpace()
        self.obj_class = self.prepare_opt_env()
        self.save_path = "history/log/{}/{}/{:0>3}/".format(opt_name, obj_name, n_experiments)
        self.lock = Lock()
        create_log_dir(self.save_path)
        self.var_names = list(self.config_space._hyperparameters.keys())

    def dict_to_list(self, hp_dict):
        """
        converting a dict of a hyperparameter configuration into a list

        Parameters
        ----------
        hp_dict: list of a hyperparameter configuration
            indexes follow the config space and values are corresponding values 
        
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

        return hp_list

    def save_hps(self, hps, ys, job_id, converted=False):
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

        if type(hps) == dict:
            hps = self.dict_to_list(hps)
        if type(ys) != dict:
            raise ValueError("ys must be dict.")
        if converted:
            hps = revert_hps(hps, self.config_space)
        for idx, hp in enumerate(hps):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, hp)

        for var_name, v in ys.items():
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, v)
    
    def load_hps(self, convert=False, do_sort=False):
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
        hps = []
        ys = []
        for idx in range(len(names)):
            var_name = names[idx]
            var_type = distribution_type(cs, var_name)
            save_file_path = self.save_path + "/" + var_name + ".csv"
            hp = load_hp(save_file_path, self.lock, var_type)
            if convert:
                hp = [convert_hp(val, cs, var_name) for val in hp]
            hps.append(hp)

        for y_name in self.y_names:
            save_file_path = self.save_path + "/" + y_name + ".csv"
            y = load_hp(save_file_path, self.lock, float)
            ys.append(y)
        
        if do_sort:
            order = np.argsort(ys[0])
            ys = [np.array(y[order]) for y in ys]
            hps = [np.array(hp[order]) for hp in hps]
        n_confs = len(hps[0])
        n_dims = len(hps)
        hps = [[hps[nd][nc] for nd in range(n_dims)] for nc in range(n_confs)]

        return hps, ys

    def prepare_opt_env(self):
        """
        The function to create ConfigSpace and load the objective function's class
        """

        with open("params.json") as f:
            json_params = json.load(f)[self.obj_name]

        config_info = json_params["config"]

        if 'dim' in json_params.keys():
            n = json_params['dim']
            v = config_info['x']
            l, u = v["lower"], v["upper"]
            for i in range(n):
                var_name = 'x{:0>3}'.format(i)
                hp = create_hyperparameter(float, var_name, lower=l, upper=u)
                self.config_space.add_hyperparameter(hp)
        else:
            for var_name, v in config_info.items():
                dist = v["dist"]
                if dist == "uniform":
                    q = None if "q" not in v.keys() or v["q"] == "None" else v["q"]
                    log = False if "log" not in v.keys() else eval(v["log"])
                    l, u = v["lower"], v["upper"]
                    vt = eval(v["var_type"])
                    hp = create_hyperparameter(vt, var_name, lower=l, upper=u, log=log, q=q)
                elif dist == "cat":
                    choices = v["choices"]
                    vt = eval(v["var_type"])
                    hp = create_hyperparameter(vt, var_name, choices=choices)
                else:
                    raise ValueError("The first element of json hp dict must be uniform or cat.")

                self.config_space.add_hyperparameter(hp)

        return load_class("obj_functions.{}.{}".format(json_params["file_name"], self.obj_name))
