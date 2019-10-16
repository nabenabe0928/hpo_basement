import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(hp_conf))])
    t1 = 1.
    t2 = 1.0 / 4000.0 * np.sum(hp_conf ** 2)
    t3 = - np.prod(np.cos(hp_conf * w))
    loss = t1 + t2 + t3
    return {"loss": loss}
