from multiprocessing import Process, Lock
import os
import csv
import time


def optimize(obj, save_file_path, max_evals=100, n_parallels=1):
    if n_parallels <= 1:
        _optimize_sequential(obj, save_file_path, max_evals=max_evals)
    else:
        _optimize_parallel(obj, save_file_path, max_evals=max_evals, n_parallels=n_parallels)


def get_n_jobs(save_file_path):
    param_files = os.isdir(save_file_path)
    n_jobs = 0
    if len(param_files) > 0:
        for param_file in param_files:
            with open(save_file_path + param_file, "r", newline="") as f:
                n_jobs = max(len(list(csv.reader(f, delimiter=","))))
    else:
        n_jobs = 0

    return n_jobs


def _optimize_sequential(obj, save_file_path, max_evals=100):
    n_jobs = get_n_jobs(save_file_path)
    lock = Lock()

    while True:
        n_gpu = 0
        obj(save_file_path, n_gpu, n_jobs, lock)
        n_jobs += 1

        if n_jobs >= max_evals:
            break


def _optimize_parallel(obj, save_file_path, max_evals=100, n_parallels=4):
    n_jobs = get_n_jobs(save_file_path)
    jobs = []
    n_runnings = 0
    lock = Lock()

    while True:
        gpus = [False for _ in range(n_parallels)]
        if len(jobs) > 0:
            n_runnings = 0
            new_jobs = []
            for job in jobs:
                if job[1].is_alive():
                    new_jobs.append(job)
                    gpus[job[0]] = True
            jobs = new_jobs
            n_runnings = len(jobs)
        else:
            n_runnings = 0

        for _ in range(max(0, n_parallels - n_runnings)):
            n_gpu = gpus.index(False)
            p = Process(target=obj, args=(save_file_path, n_gpu, n_jobs, lock))
            p.start()
            jobs.append([n_gpu, p])
            n_jobs += 1

            if n_jobs >= max_evals:
                break

            time.sleep(1.0e-6)

        if n_jobs >= max_evals:
            break
