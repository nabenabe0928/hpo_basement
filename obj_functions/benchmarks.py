import numpy as np


def ackley(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    t1 = 20
    t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(hp_conf) * np.sum(hp_conf ** 2)))
    t3 = np.e
    t4 = - np.exp(1.0 / len(hp_conf) * np.sum(np.cos(2 * np.pi * hp_conf)))
    loss = t1 + t2 + t3 + t4
    return {"loss": loss}


def different_power(hp_conf, cuda_id, save_path, experimental_settings):
    loss = 0
    for i, v in enumerate(hp_conf):
        loss += np.abs(v) ** (i + 2)

    return {"loss": loss}


def griewank(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    w = np.array([1.0 / np.sqrt(i + 1) for i in range(len(hp_conf))])
    t1 = 1.
    t2 = 1.0 / 4000.0 * np.sum(hp_conf ** 2)
    t3 = - np.prod(np.cos(hp_conf * w))
    loss = t1 + t2 + t3
    return {"loss": loss}


def k_tablet(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    k = int(np.ceil(len(hp_conf) / 4.0))
    t1 = np.sum(hp_conf[:k])
    t2 = 100 ** 2 * np.sum(hp_conf[k:] ** 2)
    loss = t1 + t2
    return {"loss": loss}


def perm(hp_conf, cuda_id, save_path, experimental_settings):
    loss = 0
    for j in range(len(hp_conf)):
        val = 0

        for i in range(len(hp_conf)):
            val += (i + 2) * (hp_conf[i] ** (j + 1) - ((1 / (i + 1)) ** (j + 1)))
        loss += val ** 2

    return {"loss": loss}


def rastrigin(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    t1 = 10 * len(hp_conf)
    t2 = np.sum(hp_conf ** 2)
    t3 = - 10 * np.sum(np.cos(2 * np.pi * hp_conf))
    loss = t1 + t2 + t3
    return {"loss": loss}


def rosenbrock(hp_conf, cuda_id, save_path, experimental_settings):
    loss = 0
    for i in range(len(hp_conf) - 1):
        t1 = 100 * (hp_conf[i + 1] - hp_conf[i] ** 2) ** 2
        t2 = (hp_conf[i] - 1) ** 2
        loss += t1 + t2
    return {"loss": loss}


def schwefel(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    loss = - np.sum(hp_conf * np.sin(np.sqrt(np.abs(hp_conf))))
    return {"loss": loss}


def sphere(hp_conf, cuda_id, save_path, experimental_settings):
    return {"loss": (np.array(hp_conf) ** 2).sum()}


def styblinski(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(hp_conf ** 4)
    t2 = - 16 * np.sum(hp_conf ** 2)
    t3 = 5 * np.sum(hp_conf)
    loss = 0.5 * (t1 + t2 + t3)
    return {"loss": loss}


def weighted_sphere(hp_conf, cuda_id, save_path, experimental_settings):
    loss = np.array([(i + 1) * hp ** 2 for i, hp in enumerate(hp_conf)])
    loss = np.sum(loss)
    return {"loss": loss}


def xin_she_yang(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(np.abs(hp_conf))
    e1 = - np.sum(np.sin(hp_conf ** 2))
    t2 = np.exp(e1)
    loss = t1 * t2
    return {"loss": loss}


def zakharov(hp_conf, cuda_id, save_path, experimental_settings):
    hp_conf = np.array(hp_conf)
    t1 = np.sum(hp_conf)
    w = np.array([i + 1 for i in range(len(hp_conf))])
    wx = np.dot(w, hp_conf)
    t2 = 0.5 ** 2 * wx ** 2
    t3 = 0.5 ** 4 * wx ** 4
    loss = t1 + t2 + t3
    return {"loss": loss}
