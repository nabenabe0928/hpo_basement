def f(hp_conf, n_gpu=None):
    loss = 0
    for i in range(len(hp_conf) - 1):
        t1 = 100 * (hp_conf[i + 1] - hp_conf[i] ** 2) ** 2
        t2 = (hp_conf[i] - 1) ** 2
        loss += t1 + t2
    return {"loss": loss}
