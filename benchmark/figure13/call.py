import pandas as pd
import numpy as np

def getdata(file):
    data = pd.read_csv(file, header=None).to_numpy()
    return np.average(data.reshape((-1, 5)), axis=1)

apps = ("pagerank-opt", "pagerank-base", "pagerank-asyn")
parts=(25, 50, 100, 150)

prefix = "result/friendster-"
suffix = "-calls.txt"

print("baseline CFP SL CFP+SL")
legend = ["iteration", "graph pool", "load graph"]
for j, p in enumerate(parts):
    print(p)

    opt = getdata(prefix + "pagerank-opt-" + str(p) + suffix)
    asyn = getdata(prefix + "pagerank-asyn-" + str(p) + suffix)
    base = getdata(prefix + "pagerank-base-" + str(p) + suffix)
    fifo = getdata(prefix + "pagerank-fifo-" + str(p) + suffix)

    for i in range(len(base)):
        print(legend[i], base[i], fifo[i], asyn[i], opt[i])

    print('hit rate', 1 - base[2] / base[0], 1 - fifo[2] / fifo[0], 1 - asyn[2] / asyn[0], 1 - opt[2] / opt[0])
