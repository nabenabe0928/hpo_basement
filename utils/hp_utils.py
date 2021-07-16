"""Unitility functions and class for hyperparameter optimization

Definition of Variables:
    N (int): The number of evaluations
    D (int): The number of dimensions
    M (int): The number of tasks (only for multi-task learning)
    K (int): The number of classes on a given task.
    yN (int): The number of types of performances recorded in an experiment.
              (not only for multi-objective optimization.)

TODO:
"""

import numpy as np
import json
import csv
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import utils
from multiprocessing import Lock
from typing import List, Union, Tuple, Any, Dict, NamedTuple
from utils.constants import ObjectiveFuncType, HyperparameterTypes, ConfigurationTypes


def create_hyperparameter(var_type: type, name: str,
                          lower: Union[int, float] = None, upper: Union[int, float] = None,
                          log: bool = False, q: Union[int, float] = None,
                          choices: List[Union[str, int, float]] = None) -> HyperparameterTypes:
    """
    Args:
        var_type (type):
            int or float or str
        name (str):
            variable name
        q (Union[float, int]):
            Discritization factor of the variable
        lower (Union[float, int]):
            lower bound of the variable
        upper (Union[float, int]):
            upper bound of the variable
        choices (List[Union[str, int, float]]):
            the choices of categorical parameter

    Returns:
        _ (HyperparameterTypes):
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


def distribution_type(config_space: CS.ConfigurationSpace, var_name: str
                      ) -> Union[str, float, int]:
    """
    Args:
        config_space (CS.ConfigurationSpace):
            Configuration space that has the searching space information
        var_name (str):
            the name of target hyperparameter.

    Returns:
        var_type (Union[str, float, int]):
            type of hyperparameter specified by `var_name`
    """

    cs_dist = str(type(config_space._hyperparameters[var_name]))

    if "Integer" in cs_dist:
        return int
    elif "Float" in cs_dist:
        return float
    elif "Categorical" in cs_dist:
        var_type = type(config_space._hyperparameters[var_name].choices[0])
        if var_type == str or var_type == bool:
            return var_type
        else:
            raise ValueError("The type of categorical parameters must be 'bool' or 'str'.")
    else:
        raise NotImplementedError("The distribution is not implemented.")


def get_hp_info(hp: HyperparameterTypes
                ) -> Tuple[Union[float, int], Union[float, int], Union[float, int], bool]:
    """
    Args:
        hp (HyperparameterTypes):
            the information of a hyperparameter

    Returns:
        lower, upper, q, log (Tuple[Union[float, int], Union[float, int], Union[float, int], bool]):
    """

    try:
        return (hp.lower, hp.upper, hp.q, hp.log) if not hp.log else \
            (np.log(hp.lower), np.log(hp.upper), hp.q, hp.log)
    except NotImplementedError:
        raise NotImplementedError("Categorical parameters do not have the log scale option.")


def save_hp(save_file_path: str, lock: Any,
            job_id: int, value: Union[int, float, str],
            record: bool = True) -> None:
    """
    recording a hyperparameter evaluated in an experiment

    Args:
        save_file_path (str):
            the path of a file to record a hyperparameter
        lock (LockType):
            preventing multiple accesses to a file
        job_id (int):
            the number of evaluations in an experiment
        value (Union[float, int, str]):
            the value of a hyperparameter evaluated in this iteration
        record (bool):
            if recording the configurations or not
    """

    if not os.path.isfile(save_file_path):
        with open(save_file_path, "w", newline="") as f:
            pass

    if record:
        lock.acquire()
        with open(save_file_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([job_id, value])
        lock.release()


def load_hps(load_file_path: str, lock: Any, var_type: type) -> np.ndarray:
    """
    loading hyperparameters evaluated in an experiment

    Args:
        load_file_path (str):
            the path of a file to load a hyperparameter
        lock (LockType):
            preventing multiple accesses to a file
        var_type: type
            the type of hyperparameter

    Returns:
        _ (np.ndarray):
            the ndarray of a hyperparameter evlauated in an experiment (N, )
            and the number of referred jobs
    """

    lock.acquire()
    with open(load_file_path, "r", newline="") as f:
        reader = list(csv.reader(f, delimiter=","))
        values = np.array([var_type(row[1]) for row in reader])
        job_id = np.array([int(row[0]) for row in reader])
    lock.release()

    order = np.argsort(job_id)

    try:
        return values[order], job_id[order[-1]]
    except IndexError:
        raise IndexError("Probably, the names of objective functions are not correct\n\
            Make sure the names of the objective functions is the same as the ones in params.json.")


class HyperparameterUtilities():
    """Utility of hyperparameter searching space

    This class allows users to easily scale each dimension

    Args:
        experimental_settings (NamedTuple):
            *func_name (str):
                The name of the objective function's file
            *dim (int):
                the dimension of a benchmark function. Required only when using a benchmark function.
            *default (bool):
                If using default hyperparameter configuration or not.
            *dataset_name (str):
                the name of dataset. e.g.) cifar, svhn etc...
            *n_cls (int):
                the number of classes on a given task.
            *image_size (int):
                pixel size
            *data_frac (float):
                How much percentages of training data to use in training. The value must be [0., 1.].
            *biased_cls (List[float]): shape -> (K, )
                How much percentages of i-th labeled data to use in training.
                The length must be same as n_cls. Each element must be (0, 1].
            *test (bool):
                If using validation dataset or test dataset. If false, using validation dataset.
    """

    def __init__(self, experimental_settings):
        """
        Attributes:
            config_space (CS.ConfigurationSpace):
                config space that contains the hyperparameter information
            obj_class (ObjectiveFuncType):
                The class of the objective function
            y_names (List[str]): shape = (yN, )
                the names of the measurements of hyperparameter configurations
            in_fmt (str):
                The format of input for the objective function. Either "dict" or "list".
            save_path (str):
                the path where we save hyperparameter configurations and corresponding performances.
                history/{log, stdo}/name of optimizer/name of algorithm/number of experiments
        """

        self.obj_name = experimental_settings.func_name
        self.config_space = CS.ConfigurationSpace()
        self.y_names = None
        self.y_upper_bounds = None
        self.in_fmt = None
        self.lock = Lock()
        self.save_path = None
        self.waiting_time = None
        self.obj_class = self.prepare_opt_env(experimental_settings)
        self.n_dimension = len(self.config_space._hyperparameters)
        self.var_names = [self.config_space._idx_to_hyperparameter[i] for i in range(self.n_dimension)]
        self.dist_types = {var_name: distribution_type(self.config_space, var_name) for var_name in self.var_names}
        self.hp_infos = {var_name: self.config_space._hyperparameters[var_name] for var_name in self.var_names}

    def dict_to_list(self, hp_dict: Dict[str, Union[float, int, str]]
                     ) -> List[Union[float, int, str]]:
        """
        converting a dict of a hyperparameter configuration into a list

        Args:
            hp_dict (Dict[str, Union[float, int, str]]):
                dict of a hyperparameter configuration
                each value will be evaluated

        Returns:
            hp_list (List[Union[float, int, str]]):
                list of a hyperparameter configuration
                each value will be evaluated
                shape is (D, ).
                The indices follow the config space
        """

        if type(hp_dict) != dict:
            raise NotImplementedError("{} is not dict.".format(hp_dict))

        hp_list = [hp_dict[var_name] for var_name in self.var_names]

        if None in hp_list:
            raise ValueError("hp_list is including None.")

        return hp_list

    def list_to_dict(self, hp_list: List[Union[float, int, str]]
                     ) -> Dict[str, Union[float, int, str]]:
        """
        converting a list of a hyperparameter configuration into a dict

        Args:
            hp_list (List[Union[float, int, str]]): shape = (D, )
                indices follow the config space and values are corresponding values

        Returns:
            hp_dict (Dict[str, Union[float, int, str]]):
                dict of a hyperparameter configuration
                each value will be evaluated
        """

        if type(hp_list) != list:
            raise NotImplementedError("{} is not list.".format(hp_list))

        hp_dict = {var_name: hp_value for hp_value, var_name in zip(hp_list, self.var_names)}

        return hp_dict

    def out_of_domain(self, hp_conf: ConfigurationTypes) -> bool:
        """
        Args:
            hp_conf (ConfigurationTypes):
                if list, the indices follow the indices in the ConfigSpace
                else if dict, the keys are the name of hyperparameters
                values are the value of hyperparameter which will be evaluated.

        Returns:
            _ (bool):
                if the value of hyperparameter is out of domain, True.
                otherwise, False.
        """

        hp_dict = self.list_to_dict(hp_conf) if type(hp_conf) == list else hp_conf

        for var_name, value in hp_dict.items():
            if not self.dist_types[var_name] in [int, float]:
                continue

            hp = self.config_space._hyperparameters[var_name]
            lb, ub = hp.lower, hp.upper

            if value < lb or ub < value:
                return True

        return False

    def pack_into_domain(self, hp_conf: ConfigurationTypes) -> Dict[str, Union[float, int, str]]:
        """
        Args:
            hp_conf (ConfigurationTypes):
                if list, the indices follow the indices in the ConfigSpace
                else if dict, the keys are the name of hyperparameters
                values are the value of hyperparameter which will be evaluated.

        Returns:
            hp_conf (Dict[str, Union[float, int, str]]):
                same type as input of this function.
                every element is bounded in the boundary described in params.json
        """

        is_conf_list = type(hp_conf) == list
        hp_dict = self.list_to_dict(hp_conf) if is_conf_list else hp_conf

        hp_dict = {var_name: np.clip(value,
                                     self.config_space._hyperparameters[var_name].lower,
                                     self.config_space._hyperparameters[var_name].upper)
                   if self.dist_types[var_name] in [int, float]
                   else value
                   for var_name, value in hp_dict.items()}

        return self.dict_to_list(hp_dict) if is_conf_list else hp_dict

    def convert_hp(self, hp_value: Union[int, float], var_name: str) -> float:
        """
        converting the value of hyperparameter in [0, 1]

        Args:
            hp_value (Union[int, float]):
                the value of a hyperparameter
            var_name (str):
                the name of a hyperparameter

        Returns:
            _ (float):
                The bounded value ([0, 1])
        """

        try:
            lb, ub, _, log = get_hp_info(self.hp_infos[var_name])
            hp_value = np.log(hp_value) if log else hp_value
            return (hp_value - lb) / (ub - lb)
        except NotImplementedError:
            raise NotImplementedError("Categorical parameters do not have lower and upper options.")

    def convert_hp_conf(self, hp_conf: List[Union[str, float, int]]) -> List[Union[str, float]]:
        """
        Converting each value of a hyperparameter configuration into [0, 1]

        Args:
            hp_conf (List[Union[str, float, int]]):
                one hyperparameter configuration

        Returns:
            _, (List[Union[str, float]]):
                the values are bounded in [0, 1].
        """

        hp_converted_conf = [self.convert_hp(hp_value, var_name)
                             if self.dist_types[var_name] in [float, int]
                             else hp_value
                             for var_name, hp_value in zip(self.var_names, hp_conf)]

        return hp_converted_conf

    def convert_hp_confs(self, hp_confs: List[List[Union[str, float, int]]]
                         ) -> List[List[Union[str, float]]]:
        """
        Converting each value of hyperparameter configurations into [0, 1]

        Args:
            hp_confs (List[List[Union[str, float, int]]]): shape = (N, D)
                the list of hyperparameter configurations

        Returns:
            _, (List[Union[str, float]]):
                the values are bounded in [0, 1].
        """

        hp_converted_confs = [self.convert_hp_conf(hp_conf) for hp_conf in hp_confs]
        return hp_converted_confs

    def revert_hp(self, hp_converted_value: float, var_name: str):
        """
        reverting the value of hyperparameter into an original scale

        Parameters
        ----------
        hp_converted_value (float):
            converted value of a hyperparameter
        var_name (str):
            the name of a hyperparameter

        Returns
            _ (Union[int, float]):
                the value in an original scale
        """

        try:
            lb, ub, q, log = get_hp_info(self.hp_infos[var_name])
            var_type = self.dist_types[var_name]
            hp_value = (ub - lb) * hp_converted_value + lb
            hp_value = np.exp(hp_value) if log else hp_value
            hp_value = np.round(hp_value / q) * q if q is not None else hp_value
            return float(hp_value) if var_type is float else int(np.round(hp_value))
        except NotImplementedError:
            raise NotImplementedError("Categorical parameters do not have lower and upper options.")

    def revert_hp_conf(self, hp_converted_conf: List[Union[str, float]]) -> List[Union[str, float, int]]:
        """
        Reverting each value of a hyperparameter configuration into original scales

        Args:
            hp_converted_conf (List[Union[str, float]]): shape = (D, )
                one hyperparameter configuration constrained in [0, 1]

        Returns:
            _ (List[Union[str, float, int]]):
                the values of a hyperparameter configuration on original scales
        """

        hp_conf = [self.revert_hp(hp_converted_value, var_name)
                   if self.dist_types[var_name] in [int, float]
                   else hp_converted_value
                   for var_name, hp_converted_value in zip(self.var_names, hp_converted_conf)]

        return hp_conf

    def revert_hp_confs(self, converted_hp_confs: List[List[Union[str, float]]]
                        ) -> List[List[Union[str, float, int]]]:
        """
        Reverting each value of hyperparameter configurations into original scales

        Args:
            converted_hp_confs (List[List[Union[str, float]]]): shape = (N, D)
                the list of hyperparameter configurations constrained in [0, 1]

        Returns:
            _ (List[List[Union[str, float, int]]]): shape = (N, D)
                the values of hyperparameter configurations in original scales
        """

        hp_confs = [self.revert_hp_conf(converted_hp_conf) for converted_hp_conf in converted_hp_confs]
        return hp_confs

    def save_hp_conf(self, hp_conf: ConfigurationTypes,
                     ys: Dict[str, Union[float, int]], job_id: int,
                     converted: bool = False, record: bool = True) -> None:
        """
        recording a hyperparameter configuration and the corresponding performance

        Args:
            hp_conf (ConfigurationTypes):
                a hyperparameter configuration (D parameters)
            ys (Dict[str, Union[float, int]]):
                the keys are the name of objective functions
                and the values are the corresponding values
                e.g.) {"loss" 0.67, "acc": 0.81}
            job_id (int):
                the number of evaluations in an experiment
            converted (bool):
                if True, reverting into original scales
            record (bool):
                if recording the configurations or not
        """

        if type(hp_conf) == dict:
            hp_conf = self.dict_to_list(hp_conf)
        if type(ys) != dict:
            raise ValueError("ys must be dict.")
        if converted:
            hp_conf = self.revert_hp_conf(hp_conf)
        for var_name, hp in zip(self.var_names, hp_conf):
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, hp, record=record)

        for var_name, v in ys.items():
            save_file_path = self.save_path + "/" + var_name + ".csv"
            save_hp(save_file_path, self.lock, job_id, v, record=record)

    def load_hps_conf(self, convert: bool = False, do_sort: bool = False,
                      index_from_conf: bool = True, another_src: str = None
                      ) -> Tuple[List[List[Union[str, float, int]]], List[List[Union[float, int]]]]:
        """
        loading hyperparameter configurations and the corresponding performance

        Args:
            convert: bool
                if True, converting into constrained scales.
            do_sort: bool
                if True, sort configurations in ascending order by loss values.
            another_src: str
                If any path is given, configurations will be loaded from the given path.
            index_from_conf: bool
                If True, the index of hp_conf becomes opposite.
                hp_conf: [the index for hyperparameters][the index for configurations]

        Returns:
            the list of hyperparameter configurations and the corresponding performance
            hp_conf: list if index_from_conf (N, D) else (D, N)
            ys: list (yN, N)
        """

        n_referred_jobs = np.inf
        hps_conf = []
        ys = []
        for var_name in self.var_names:
            var_type = self.dist_types[var_name]
            load_file_path = self.save_path + "/" + var_name + ".csv" if another_src is None \
                else another_src + "/" + var_name + ".csv"
            hps, max_job_id = load_hps(load_file_path, self.lock, var_type)
            n_referred_jobs = min(n_referred_jobs, max_job_id)

            if convert and var_type in [float, int]:
                hps = [self.convert_hp(hp, var_name) for hp in hps]
            hps_conf.append(hps)

        for y_name in self.y_names:
            load_file_path = self.save_path + "/" + y_name + ".csv" if another_src is None \
                else another_src + "/" + y_name + ".csv"
            y, max_job_id = load_hps(load_file_path, self.lock, float)
            n_referred_jobs = min(n_referred_jobs, max_job_id)
            ys.append(np.array(y))

        hps_conf = [hps[:n_referred_jobs + 1] for hps in hps_conf]
        ys = [y[:n_referred_jobs + 1] for y in ys]

        if do_sort:
            order = np.argsort(ys[0])
            ys = [np.minimum(y[order], y_upper_bound) for y, y_upper_bound in zip(ys, self.y_upper_bounds)]
            hps_conf = [np.array(hps)[order] for hps in hps_conf]
        if index_from_conf:
            n_confs = len(hps_conf[0])
            n_dims = len(hps_conf)
            hps_conf = [[hps_conf[nd][nc] for nd in range(n_dims)] for nc in range(n_confs)]

        return hps_conf, ys

    def load_transfer_hps_conf(self, transfer_info_paths, convert=False, do_sort=False, index_from_conf=True):
        """
        loading hyperparameter configurations of prior information and the corresponding performance

        Parameters
        ----------
        transfer_info_paths: list of string (M, )
            The path of the location of prior information

        Returns
        -------
        the list of hyperparameter configurations and the corresponding performance
        X: list (M, :, D) (: is the index for configurations)
        Y: list (M, yN, :)
        """

        X = [[]]
        Y = [[]]

        if transfer_info_paths is None:
            raise ValueError("transfer_info_paths has to be list of paths of information to transfer, "
                             "but None was given.")

        for path in transfer_info_paths:
            print("### Transferring from {} ###".format(path))
            _X, _Y = self.load_hps_conf(convert=convert,
                                        do_sort=do_sort,
                                        another_src=path,
                                        index_from_conf=index_from_conf)
            X.append(_X)
            Y.append(_Y)
        return X, Y

    def prepare_opt_env(self, experimental_settings: NamedTuple) -> ObjectiveFuncType:
        """The function to create ConfigSpace and load the objective function's class"""

        with open("params.json") as f:
            json_params = json.load(f)[self.obj_name]

        config_info = json_params["config"]
        self.y_names = json_params["y_names"]
        self.y_upper_bounds = json_params["y_upper_bounds"] if "y_upper_bounds" in json_params.keys() \
            else [1.0e+8 for _ in range(len(self.y_names))]
        self.in_fmt = json_params["in_fmt"]
        self.waiting_time = json_params["waiting_time"]

        if len(self.y_names) != len(self.y_upper_bounds):
            raise ValueError("The shape of y_names and y_upper_bounds in params.json must be same.")

        if experimental_settings.dim is not None:
            try:
                v = config_info['x']
            except ValueError:
                raise ValueError("'x' is allowed to be the name of hyperparameters of benchmark functions "
                                 "ONLY in params.json.")
            lb, ub = v["lower"], v["upper"]
            for i in range(experimental_settings.dim):
                var_name = 'x{:0>5}'.format(i)
                hp = create_hyperparameter(float, var_name, lower=lb, upper=ub)
                self.config_space.add_hyperparameter(hp)
        else:
            if 'x' in config_info.keys():
                raise ValueError("-dim is required in the command line e.g.) 'python main.py -dim 3 -ini 3'"
                                 " when optimizing benchmark functions.")
            for var_name, v in config_info.items():
                if "ignore" in v.keys():
                    if v["ignore"] != "True" and v["ignore"] != "False":
                        raise ValueError("ignore must be 'True' or 'False', but {} was given.".format(v["ignore"]))
                    elif v["ignore"] == "True":
                        continue

                hp = self.get_hp_info_from_json(v, var_name)
                self.config_space.add_hyperparameter(hp)

        return utils.load_class("obj_functions.{}".format(self.obj_name))(experimental_settings)

    def get_hp_info_from_json(self, v: Dict[str, Any], var_name: str) -> HyperparameterTypes:
        """
        Args:
            v (Dict[str, Any]):
                loaded from params.json
            var_name (str):
                the name of a hyperparameter

        Returns:
            hp (HyperparameterTypes):
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
            if "True" in choices:
                choices = [eval(choice) for choice in choices]
            hp = create_hyperparameter(vt, var_name, choices=choices)
        else:
            raise ValueError("The first element of json hp dict must be 'u' (uniform) or 'c' (categorical).")

        return hp
