import numpy as np


def f(hp_conf, n_gpu=None):
    loss = np.array([(i + 1) * hp ** 2 for i, hp in enumerate(hp_conf)])
    loss = np.sum(loss)
    return {"loss": loss}
