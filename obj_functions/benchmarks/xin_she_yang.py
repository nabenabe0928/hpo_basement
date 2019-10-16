import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(np.abs(hp_conf))
    e1 = - np.sum(np.sin(hp_conf ** 2))
    t2 = np.exp(e1)
    loss = t1 * t2
    return {"loss": loss}
