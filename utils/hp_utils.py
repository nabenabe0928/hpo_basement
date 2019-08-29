import numpy as np
import json
import csv
import os
from util import load_class, create_log_dir
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def create_hyperparameter(var_type, name, lower=None, upper=None, log=False, q=None, choices=None):
    """
    options for hyperparameters:
        dist: "uniform", "cat"
        var_type: int or float or str
        q: any positive integer
        lower, upper: float or integer
        choices: str, float, integer
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
    if hp.log:
        return hp.lower, hp.upper
    else:
        return np.log(hp.lower), np.log(hp.upper)


def out_of_domain(hps, hpu):
    hp_dict = hpu.list_to_dict(hps) if type(hps) == list else hps

    for var_name, value in hp_dict.items():
        hp = hpu.config_space._hyperparameters[var_name]
        l, u = hp.lower, hp.upper
        if l <= value <= u:
            pass
        else:
            return True
    return False


def convert_hp(hp_value, cs, var_name):
    l, u = get_hp_info(cs._hyperparameters[var_name])
    return (hp_value - l) / (u - l)


def convert_hps(hp_values, cs):
    hp_converted_values = []
    for idx, hp_value in enumerate(hp_values):
        var_name = cs._idx_to_hyperparameter[idx]
        if distribution_type(cs, var_name) is not str:
            hp_converted_values.append(convert_hp(hp_value, cs, var_name))
        else:
            hp_converted_values.append(hp_value)
    return hp_converted_values


def convert_hps_set(hps_set, cs):
    hp_converted_values_set = []
    for hps in hps_set:
        hp_converted_values_set.append(convert_hps(hps, cs))
    return hp_converted_values_set


def revert_hp(hp_converted_value, cs, var_name):
    l, u = get_hp_info(cs._hyperparameters[var_name])
    var_type = distribution_type(cs, var_name)
    return var_type((u - l) * hp_converted_value + l)


def revert_hps(hp_converted_values, cs):
    hp_values = []
    for idx, hp_converted_value in enumerate(hp_converted_values):
        var_name = cs._idx_to_hyperparameter[idx]
        if distribution_type(cs, var_name) is not str:
            hp_values.append(revert_hp(hp_converted_value, cs, var_name))
        else:
            hp_values.append(hp_converted_value)

    return hp_values


def revert_hps_set(converted_hps_set, cs):
    hp_values_set = []
    for converted_hps in converted_hps_set:
        hp_values_set.append(revert_hps(converted_hps, cs))
    return hp_values_set


def save_hp(save_file_path, lock, job_id, value):
    if not os.path.isfile(save_file_path):
        with open(save_file_path, "w", newline="") as f:
            pass

    lock.acquire()
    with open(save_file_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([job_id, value])
    lock.release()


class HyperparameterUtilities():
    """
    obj_name: The name of the objective function's file
    max_evals: The maximum number of evaluations throughout an experiment
    n_parallels: The number of computer resoures used in an experiment
    config_space: ConfigurationSpace
    obj_class: The class of the objective function

    the path where we save log file or standard output.
    history/{log, stdo}/name of optimizer/name of algorithm/number of experiments
    """
    def __init__(self, obj_name, opt_name, max_evals, n_experiments, n_parallels=1):
        self.obj_name = obj_name
        self.max_evals = max_evals
        self.n_parallels = n_parallels
        self.config_space = CS.ConfigurationSpace()
        self.obj_class = self.create_config_space()
        self.save_path = "history/log/{}/{}/{:0>3}/".format(opt_name, obj_name, n_experiments)
        create_log_dir(self.save_path)

    def dict_to_list(self, hp_dict):
        hp_list = [None for _ in range(len(hp_dict))]

        for var_name, value in hp_dict.items():
            idx = self.config_space._hyperparameter_idx[var_name]
            hp_list[idx] = value

        if None in hp_list:
            raise ValueError("hp_list is including None.")

        return hp_list

    def list_to_dict(self, hp_list):
        hp_dict = {}

        for idx, value in enumerate(hp_list):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            hp_dict[var_name] = value

        return hp_list

    def save_hps(self, hps, ys, job_id, lock, converted=False):
        if type(hps) == dict:
            hps = self.dict_to_list(hps)
        if type(ys) != dict:
            raise ValueError("ys must be dict.")
        if converted:
            hps = revert_hps(hps, self.config_space)
        for idx, hp in enumerate(hps):
            var_name = self.config_space._idx_to_hyperparameter[idx]
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, lock, job_id, hp)

        for var_name, v in ys.items():
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, lock, job_id, v)

    def create_config_space(self):
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
                    q = None if "q" not in v.keys() else v["q"]
                    log = None if "log" not in v.keys() else v["log"]
                    l, u = v["lower"], v["upper"]
                    vt = v["var_type"]
                    hp = create_hyperparameter(vt, var_name, lower=l, upper=u, log=log, q=q)
                elif dist == "cat":
                    choices = v["choices"]
                    vt = v["var_type"]
                    hp = create_hyperparameter(vt, var_name, choices=choices)
                else:
                    raise ValueError("The first element of json hp dict must be uniform or cat.")

                self.config_space.add_hyperparameter(hp)

        return load_class("obj_functions.{}.{}".format(json_params["file_name"], self.obj_name))
