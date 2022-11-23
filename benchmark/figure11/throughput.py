import pandas as pd
import numpy as np

def get_data(app, graph):
    data = pd.read_csv("result/" + graph + "-" + app + "-t.txt", header=None).to_numpy()
    data = data.reshape((-1, 6)) / 1000
    data = data[:,0]
    print(np.std(data) / np.mean(data))
    return data

apps = ["genericwalk", "ppr"]
datasets = ["livejournal", "orkut", "twitter"]

V = np.array([4846609, 3072441, 41652230])
steps = {"genericwalk": V * 2 * 80, "pagerank": V * 2 * 80, "ppr": V * 2 / 0.15}

with open("lighttraffic_throughput_small.txt", "w") as f:
    for app in apps:
        for i, g in enumerate(datasets):
            data = get_data(app, g)
            for t in data:
                f.writelines(str(steps[app][i] / t) + '\n')
