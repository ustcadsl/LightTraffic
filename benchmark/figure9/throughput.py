import pandas as pd
import numpy as np

def get_data(dir, app, graph):
    data = pd.read_csv(dir + "/" + graph + "-" + app + "-t.txt", header=None).to_numpy()
    data = data.reshape((-1, 6)) / 1000
    data = data[:,0]
    print(np.std(data) / np.mean(data))
    return data

apps = ["genericwalk", "pagerank", "ppr"]
datasets = ["friendster", "uk-union", "yahoo", "clueweb"]

V = np.array([68349466, 131572430, 653912704, 1684868322])
steps = {"genericwalk": V * 2 * 80, "pagerank": V * 2 * 80, "ppr": V * 2 / 0.15}

def print_throughput(dir, name):
    with open(name + "_throughput.txt", "w") as f:
        for app in apps:
            for i, g in enumerate(datasets):
                data = get_data(dir, app, g)
                for t in data:
                    f.writelines(str(steps[app][i] / t) + '\n')

print_throughput("result", "lighttraffic")
print_throughput("pcie3", "pcie3")
print_throughput("pcie4", "pcie4")
