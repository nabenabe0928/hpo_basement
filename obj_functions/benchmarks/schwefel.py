import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    loss = - np.sum(hp_conf * np.sin(np.sqrt(np.abs(hp_conf))))
    return {"loss": loss}
