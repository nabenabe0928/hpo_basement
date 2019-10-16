import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    t1 = 20
    t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(hp_conf) * np.sum(hp_conf ** 2)))
    t3 = np.e
    t4 = - np.exp(1.0 / len(hp_conf) * np.sum(np.cos(2 * np.pi * hp_conf)))
    loss = t1 + t2 + t3 + t4
    return {"loss": loss}
