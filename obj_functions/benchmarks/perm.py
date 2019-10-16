def f(hp_conf, n_gpu=None):
    loss = 0
    for j in range(len(hp_conf)):
        val = 0

        for i in range(len(hp_conf)):
            val += (i + 2) * (hp_conf[i] ** (j + 1) - ((1 / (i + 1)) ** (j + 1)))
        loss += val ** 2

    return {"loss": loss}
