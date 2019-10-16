import numpy as np


def f(hp_conf, n_gpu=None):
    loss = 0
    for i, v in enumerate(hp_conf):
        loss += np.abs(v) ** (i + 2)

    return {"loss": loss}
