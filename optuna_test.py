import optuna
import numpy as np


def obj(trial):
    xs = np.array([trial.suggest_uniform("x{}".format(i), -5, 5) for i in range(10)])
    return (xs ** 2).sum()


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(obj, n_trials=100)
    # 20:08:29 - 20:11:01 -> 152[s] vs 8.95[s]
