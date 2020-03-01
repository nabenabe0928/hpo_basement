import sys
import os
import csv
import subprocess as sp
import time
import optimizer
import obj_functions
from argparse import ArgumentParser
from typing import NamedTuple


class pycolor:
    RED = '\033[31m'
    YELLOW = '\033[33m'
    END = '\033[0m'


"""
Parameters
----------
func_name: str
    The name of callable you want to optimize.
dim: int
    the dimension of a benchmark function. Required only when using a benchmark function.
dataset_name: string
    the name of dataset. e.g.) cifar, svhn etc...
n_cls: int
    the number of classes on a given task.
image_size: int
    pixel size
data_frac: float
    How much percentages of training data to use in training. The value must be [0., 1.].
biased_cls: list of float (K, )
    How much percentages of i-th labeled data to use in training.
    The length must be same as n_cls. Each element must be (0, 1].
test: bool
    If using validation dataset or test dataset. If false, using validation dataset.
extra_opt_name: str
    The information added to optimizer name. (Used only for the specification of path to save log.)
extra_exp_name: str
    The information added to experiment's name. (Used only for the specification of path to save log.)
"""


class ExperimentalSettings(
    NamedTuple("_ExperimentalSettings",
               [("func_name", str),
                ("dim", int),
                ("dataset_name", str),
                ("n_cls", int),
                ("image_size", int),
                ("data_frac", float),
                ("biased_cls", list),
                ("test", bool),
                ("all_train", bool),
                ("extra_opt_name", str),
                ("extra_exp_name", str)
                ])):
    pass


def print_parser_warning():
    print("#### PARSER ERROR ####")
    print("One example to run the file is described below:")
    print("")
    print("user@user:~$ python main.py -fuc sphere -dim 2 -par 1 -ini 10 -exp 0 -eva 100 \
-res 0 [-seed 0 -veb 1 -fre 1 -dat cifar -cls 10 -img 32 -sub 0.1 -defa 0 -che 0]")
    print("")
    print("  -fuc (Both  Required, default: None): The name of callable you would like to optimize.")
    print("  -ini (Both  Required, default: None): The number of initial samplings.")
    print("  -tra (Trans Required, default; []  ): The list of the path of previous information to transfer. 'opt/function/number'")
    print("  -dim (Bench Required, default: None): The dimension of a hyperparameter configuraiton. (Only for benchmark functions. Otherwise, omit it.)")
    print("  -dat (ML    Required, default: None): The name of dataset.")
    print("  -cls (ML    Required, default: None): The number of classes on a given task.")
    print("  -img (ML    Optional, default: None): The pixel size of training data. (If None, using the possible maximum pixel size.)")
    print("  -sub (ML    Optional, default: 1.  ): How much percentages of training data to use in training (Must be between 0. and 1.).")
    print("  -defa(ML    Optional, default: 0   ): If using the default hyperparameter configuration or not (If 1, using the default configuration.).")
    print("  -test(ML    Optional, default: 0   ): If using validation dataset or test dataset. (If 1, using test dataset.).")
    print("  -altr(ML    Optional, default: 0   ): If using all the training data or not. (If 1, using all the data.).")
    print("  -cuda(ML    Optional, default: range(par) ): Which CUDA devices you use in the experiment. (Specify the single or multiple number(s))")
    print("  -par (Both  Optional, default: 1   ): The number of parallel computer resources.")
    print("  -exp (Both  Optional, default: 0   ): The index of an experiment. (Used only to specify the path of log files.)")
    print("  -eva (Both  Optional, default: 100 ): The number of evaluations in an experiment.")
    print("  -res (Both  Optional, default: 0   ): Whether restarting the previous experiment or not. If 0, removes the previous log files.")
    print("  -seed(Both  Optional, default: None): The number to specify a random number generator.")
    print("  -bar (Both  Optional, default: 1   ): Whether to use the barrier function or not. if 0, make hyperparameter get in feasible domain.")
    print("  -veb (Both  Optional, default: 1   ): Whether print the result or not. If 0, do not print.")
    print("  -fre (Both  Optional, default: 1   ): Every print_freq iteration, the result will be printed.")
    print("  -che (Both  Optional, default: 1   ): If asking when removing files or not at the initialization.")
    print("  -eopt(Both  Optional, default: None): The information added to optimizer name. (Used only for the specification of path to save log.)")
    print("  -eexp(Both  Optional, default: None): The information added to experiment's name. (Used only for the specification of path to save log.)")
    print("")
    sys.exit()


def parse_requirements():
    import subprocess
    try:
        msg = subprocess.check_output("nvidia-smi --query-gpu=index --format=csv", shell=True)
        n_devices = max(0, len(msg.decode().split("\n")) - 2)
    except subprocess.CalledProcessError:
        n_devices = 0

    ap = ArgumentParser()
    ap.add_argument("-fuc", type=str, default=None)
    ap.add_argument("-ini", type=int, default=None)
    ap.add_argument("-dim", type=int, default=None)
    ap.add_argument("-dat", type=str, default=None)
    ap.add_argument("-cls", type=int, default=None)
    ap.add_argument("-img", type=int, default=None)
    ap.add_argument("-sub", type=float, default=None)
    ap.add_argument("-defa", type=int, choices=[0, 1], default=0)
    ap.add_argument("-test", type=int, choices=[0, 1], default=0)
    ap.add_argument("-altr", type=int, choices=[0, 1], default=0)
    ap.add_argument("-cuda", type=int, nargs="*", choices=range(n_devices), default=[0])
    ap.add_argument("-par", type=int, default=1)
    ap.add_argument("-exp", type=int, default=0)
    ap.add_argument("-eva", type=int, default=100)
    ap.add_argument("-res", type=int, choices=[0, 1], default=0)
    ap.add_argument("-seed", type=int, default=None)
    ap.add_argument("-veb", type=int, choices=[0, 1], default=1)
    ap.add_argument("-bar", type=int, choices=[0, 1], default=1)
    ap.add_argument("-fre", type=int, default=1)
    ap.add_argument("-che", type=int, choices=[0, 1], default=1)
    ap.add_argument("-tra", type=str, nargs="*", default=[])
    ap.add_argument("-eopt", type=str, default=None)
    ap.add_argument("-eexp", type=str, default=None)

    args = ap.parse_args()

    requirements = {"n_parallels": args.par,
                    "n_init": args.ini,
                    "n_experiments": args.exp,
                    "max_evals": args.eva,
                    "restart": bool(args.res),
                    "seed": args.seed,
                    "is_barrier": bool(args.bar),
                    "verbose": bool(args.veb),
                    "print_freq": args.fre,
                    "default": bool(args.defa),
                    "check": bool(args.che),
                    "cuda": args.cuda if len(args.cuda) == args.par else list(range(args.par)),
                    "transfer_info_paths": ["history/log/{}".format(path) for path in args.tra]
                    }
    if len(requirements["transfer_info_paths"]) == 0:
        requirements["transfer_info_paths"] = None

    if args.ini is None or args.fuc is None:
        print("The list of func_name is as follows:")
        print(obj_functions.__all__[obj_functions.n_non_func:])
        print("")
        print_parser_warning()

    if args.dim is not None and args.dat is not None:
        print("dim and dat cannot coexist.")
        print_parser_warning()
    elif args.altr and not args.test:
        print("if using all the training data, you have to use test dataset.")
        print_parser_warning()
    elif args.dim is not None:
        experimental_settings = {"func_name": args.fuc,
                                 "dim": args.dim,
                                 "dataset_name": None,
                                 "n_cls": None,
                                 "image_size": None,
                                 "data_frac": None,
                                 "biased_cls": None,
                                 "test": None,
                                 "all_train": None
                                 }
    elif args.dat is not None and args.cls is not None:
        experimental_settings = {"func_name": args.fuc,
                                 "dim": None,
                                 "dataset_name": args.dat,
                                 "n_cls": args.cls,
                                 "image_size": args.img,
                                 "data_frac": args.sub,
                                 "biased_cls": None,
                                 "test": bool(args.test),
                                 "all_train": bool(args.altr)
                                 }
    else:
        print("### Check the requirements for the running command below ###")
        print_parser_warning()

    experimental_settings["extra_opt_name"] = args.eopt
    experimental_settings["extra_exp_name"] = args.eexp

    return optimizer.BaseOptimizerRequirements(**requirements), ExperimentalSettings(**experimental_settings)


def get_stdo_path(path):
    stdo = "history/stdo/"

    for p in path.split("/")[2:]:
        stdo += p + "/"
    return stdo


def print_result(hp_conf, ys, job_id, list_to_dict):
    print("##### Evaluation {:0>5} #####".format(job_id))
    if type(hp_conf) is dict:
        print(hp_conf)
    else:
        print(list_to_dict(hp_conf))
    print(ys)
    print("")


def save_elapsed_time(save_path, lock, verbose=True, print_freq=1):
    save_path = save_path + "/TIME.csv"
    start_time = time.time()

    if not os.path.isfile(save_path):
        lock.acquire()
        with open(save_path, "w", newline=""):
            pass
        lock.release()
    else:
        lock.acquire()
        with open(save_path, "r", newline="") as f:
            reader = list(csv.reader(f, delimiter=","))
            last_time = float(reader[-1][-1])
        lock.release()
        start_time -= last_time

    def _imp(eval_start, lock, n_jobs):
        current_time = time.time()
        eval_time = current_time - eval_start
        elapsed_time = current_time - start_time

        lock.acquire()
        if not os.path.isfile(save_path):
            with open(save_path, "w", newline=""):
                pass

        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([eval_time, elapsed_time])

        if verbose and n_jobs % print_freq == 0:
            print("Evaluation time : {:.4f}[s]".format(eval_time))
            print("Elapsed    time : {:.4f}[s]".format(elapsed_time))
            print("\n")

        lock.release()

    return _imp


def check_conflict(path, check=True):
    stdo = get_stdo_path(path)

    n_files = len(os.listdir(stdo)) + len(os.listdir(path))

    if n_files > 0:
        print("")
        print("########## CAUTION #########")
        print(pycolor.RED + "You are going to remove {} files in {} and {}.".format(n_files, path, stdo) + pycolor.END)
        print("")

        if check:
            answer = ""
            while answer not in {"y", "n"}:
                print("")
                answer = input("Is it okay? [y or n] : ")
            if answer == "y":
                pass
            else:
                print("Permission Denied.")
                sys.exit()

        sp.call("rm -r {}".format(stdo), shell=True)
        sp.call("rm -r {}".format(path), shell=True)
        print("")
        print("############################")
        print("##### REMOVE THE FILES #####")
        print("############################")
        print("")


def create_log_dir(path_log):
    """
    Parameters
    ----------
    path: string
        the path where creating the log file for hyperparameter configurations and corresponding performances.
    """

    stdo = get_stdo_path(path_log)

    for path in [stdo, path_log]:
        files = path.split("/")
        this_path = ""

        for f in files:
            this_path += f + "/"
            if not os.path.isdir(this_path):
                try:
                    os.mkdir(this_path)
                except OSError:
                    pass


def _resolve_name(path, package, start):
    """Return the absolute name of the module to be imported."""

    if not hasattr(package, 'rindex'):
        raise ValueError("'package' not set to a string")
    dot = len(package)
    for _ in range(start, 1, -1):
        try:
            dot = package.rindex('.', 0, dot)
        except ValueError:
            raise ValueError("attempted relative import beyond top-level "
                             "package")
    return "{}.{}".format(package[:dot], path)


def import_module(path, package=None):
    """Import a module.

    The 'package' argument is required when performing a relative import. It
    specifies the package to use as the anchor point from which to resolve the
    relative import to an absolute import.

    """
    if path.startswith('.'):
        if not package:
            raise TypeError("Relative imports require the 'package' argument")
        start = 0
        while path[start] == "." or start < len(path):
            start += 1
        path = _resolve_name(path[start:], package, start)
    __import__(path)

    return sys.modules[path]


def load_class(path):
    dot_loc = path.rindex(".")
    module_path, class_name = path[:dot_loc], path[dot_loc + 1:]
    module = import_module(module_path)
    return getattr(module, class_name)
