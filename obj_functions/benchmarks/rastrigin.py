import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    t1 = 10 * len(hp_conf)
    t2 = np.sum(hp_conf ** 2)
    t3 = - 10 * np.sum(np.cos(2 * np.pi * hp_conf))
    loss = t1 + t2 + t3
    return {"loss": loss}
