import sys
import os
import subprocess as sp
from argparse import ArgumentParser


class pycolor:
    RED = '\033[31m'
    YELLOW = '\033[33m'
    END = '\033[0m'


def parse_requirements():
    ap = ArgumentParser()
    ap.add_argument("-dim", type=int, default=None)
    ap.add_argument("-par", type=int)
    ap.add_argument("-ini", type=int)
    ap.add_argument("-exp", type=int)
    ap.add_argument("-eva", type=int)
    ap.add_argument("-res", type=int, choices=[0, 1])

    args = ap.parse_args()
    requirements = {"n_parallels": args.par,
                    "n_init": args.ini,
                    "n_experiments": args.exp,
                    "max_evals": args.eva,
                    "restart": args.res
                    }
    dim = args.dim

    if None in requirements.values():
        print("#### PARSER ERROR ####")
        print("One example to run the file is described below:")
        print("")
        print("python main.py -dim 2 -par 1 -ini 10 -exp 0 -eva 100 -res 0")
        print("  -dim: The dimension of a hyperparameter configuraiton. (Only for benchmark functions. Otherwise, omit it.)")
        print("  -par: The number of parallel computer resources.")
        print("  -ini: The number of initial samplings.")
        print("  -exp: The index of an experiment. (Used only to specify the path of log files.)")
        print("  -eva: The number of evaluations in an experiment.")
        print("  -res: Whether restarting the previous experiment or not. If 0, removes the previous log files.")
        print("")
        sys.exit()

    return requirements, dim


def get_stdo_path(path):
    stdo = "history/stdo/"

    for p in path.split("/")[2:]:
        stdo += p + "/"
    return stdo


def check_conflict(path):
    stdo = get_stdo_path(path)

    n_files = len(os.listdir(stdo)) + len(os.listdir(path))

    if n_files > 0:
        print("")
        print("#### CAUTION ###")
        print(pycolor.RED + "You are going to remove {} files in {} and {}.".format(n_files, path, stdo) + pycolor.END)
        print("")

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
        print("#########################")
        print("### REMOVED THE FILES ###")
        print("#########################")
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
                os.mkdir(this_path)


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
