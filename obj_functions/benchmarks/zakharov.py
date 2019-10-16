import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(hp_conf)
    w = np.array([i + 1 for i in range(len(hp_conf))])
    wx = np.dot(w, hp_conf)
    t2 = 0.5 ** 2 * wx ** 2
    t3 = 0.5 ** 4 * wx ** 4
    loss = t1 + t2 + t3
    return {"loss": loss}
