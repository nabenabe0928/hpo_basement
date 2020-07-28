import csv
import json
import numpy as np


def get_parameter(file_name, dtype=float, n=100):
    with open(file_name, "r", newline="") as f:
        reader = list(csv.reader(f, delimiter=","))
        ids = np.array([int(r[0]) for r in reader if int(r[0]) < n])
        params = np.array([dtype(r[1]) for r in reader if int(r[0]) < n])
    order = np.argsort(ids)
    return ids[order], params[order]


def get_parameterset(path, param_dict, n=100):
    paramsets = [{} for _ in range(n)]
    path = path if path[:-1] != "/" else path[:-1]
    for param_name, dt in param_dict.items():
        ids, params = get_parameter("{}/{}.csv".format(path, param_name), dt, n)
        for id, param in zip(ids, params):
            paramsets[id][param_name] = param
    return paramsets


if __name__ == "__main__":
    obj = "cnn"
    with open("params.json") as f:
        param_dict = {k: eval(v) for k, v in json.load(f)[obj].items()}
    fn = "log/RandomSearch/cnn_cifar10_transfers_RandomSearch_on_cnn_svhn10/000/"
    print(get_parameterset(fn, param_dict, 100))
