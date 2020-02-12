import numpy as np
import matplotlib.pyplot as plt
import csv


if __name__ == "__main__":
    funcs = ["griewank_10d",
             "k_tablet_10d",
             "michalewicz_10d",
             "schwefel_10d",
             "sphere_10d",
             "styblinski_10d"]
    head = "history/log/"
    tar = "/loss.csv"
    n = 500
    e = 1

    for func in funcs:
        mvs, uvs = [], []
        for i in range(e):
            mvtpe, tpe = [np.inf], [np.inf]
            with open("{}{}{}/00{}{}".format(head, "SingleTaskMultivariateTPE/", func, i, tar), "r", newline="") as f:
                reader = csv.reader(f, delimiter=",")
                for r in reader:
                    mvtpe.append(min(mvtpe[-1], float(r[1])))
            del mvtpe[0]

            with open("{}{}{}/00{}{}".format(head, "SingleTaskUnivariateTPE/", func, i, tar), "r", newline="") as f:
                reader = csv.reader(f, delimiter=",")
                for r in reader:
                    tpe.append(min(tpe[-1], float(r[1])))
            del tpe[0]
        mvs.append(mvtpe[:n])
        uvs.append(tpe[:n])
        print(func)

        uvs, mvs = map(np.asarray, [uvs, mvs])
        x = np.arange(n)
        # tpe, mvtpe = np.log(tpe), np.log(mvtpe)
        plt.title(func)
        m = uvs.mean(axis=0)
        s = uvs.std(axis=0)
        plt.plot(x, m, label="TPE", color="blue")
        plt.fill_between(x, m - s, m + s, color="blue", alpha=0.2)
        
        m = mvs.mean(axis=0)
        s = mvs.std(axis=0)
        plt.plot(x, m, label="MV-TPE", color="red")
        plt.fill_between(x, m - s, m + s, color="red", alpha=0.2)
        plt.legend()
        plt.grid()
        plt.show()
