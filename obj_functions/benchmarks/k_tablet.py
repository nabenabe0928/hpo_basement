import numpy as np


def f(hp_conf, n_gpu=None):
    hp_conf = np.array(hp_conf)
    k = int(np.ceil(len(hp_conf) / 4.0))
    t1 = np.sum(hp_conf[:k])
    t2 = 100 ** 2 * np.sum(hp_conf[k:] ** 2)
    loss = t1 + t2
    return {"loss": loss}
