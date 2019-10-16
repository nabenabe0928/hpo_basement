import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(hp_conf ** 4)
    t2 = - 16 * np.sum(hp_conf ** 2)
    t3 = 5 * np.sum(hp_conf)
    loss = 0.5 * (t1 + t2 + t3)
    return {"loss": loss}
