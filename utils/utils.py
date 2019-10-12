import sys
import os


def create_log_dir(path):
    """
    Parameters
    ----------
    path: string
        the path where creating the log file for hyperparameter configurations and corresponding performances.
    """

    files = path.split("/")
    this_path = ""

    for f in files:
        this_path += f + "/"
        if not os.path.isdir(f):
            os.mkdir(f)


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
