import numpy as np


def f(hp_conf, n_gpu=None):
    return {"loss": (np.array(hp_conf) ** 2).sum()}
