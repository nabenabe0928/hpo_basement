import pstats

if __name__ == "__main__":
    sts = pstats.Stats("hoge")
    sts.strip_dirs().sort_stats("tottime").print_stats(50)
