import numpy as np
import matplotlib.pyplot as plt
import csv


if __name__ == "__main__":
    funcs = {"griewank_10d": 0.0,
             "k_tablet_10d": 0.0,
             "michalewicz_10d": -9.66015,
             "sphere_10d": 0.0,
             "schwefel_10d": -4189.829,
             "styblinski_10d": -391.66165,
             "weighted_sphere_10d": 0.0}
    opts = {"SingleTaskMultivariateTPE/": "MV-TPE",
            "SingleTaskUnivariateTPE/": "TPE",
            "NelderMead/": "NM",
            "CMA/": "CMA",
            "LatinHypercubeSampling/": "LHS",
            "SingleTaskGPBO/": "GP-EI"}
    cs = ["red", "blue", "green", "purple", "black", "olive"]
    head = "history/log/"
    tar = "/loss.csv"
    n = 500
    is_log = True
    e = 10

    for func, min_val in funcs.items():
        res = {v: [] for v in opts.values()}
        for i in range(e):
            for opt, key in opts.items():
                ys = [np.inf]
                with open("{}{}{}/00{}{}".format(head, opt, func, i, tar), "r", newline="") as f:
                    reader = csv.reader(f, delimiter=",")
                    for r in reader:
                        ys.append(min(ys[-1], float(r[1])))
                del ys[0]
                res[key].append(ys[:n])

        print(func)

        res = {k: np.array(v) for k, v in res.items()}
        x = np.arange(n)
        # tpe, mvtpe = np.log(tpe), np.log(mvtpe)
        plt.title(func)

        for c, (k, v) in zip(cs, res.items()):
            # print(v.min())
            v = np.log(v - min_val + 1.0e-12) if is_log else v
            m = v.mean(axis=0)
            s = v.std(axis=0) / np.sqrt(e)
            plt.plot(x, m, label=k, color=c)
            plt.fill_between(x, m - s, m + s, color=c, alpha=0.2)
        
        plt.legend()
        plt.grid()
        plt.show()
