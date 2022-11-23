import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def get_total(app, graph):
    data = pd.read_csv("result/" + graph + "-" + app + "-t.txt", header=None).to_numpy()
    data = data.reshape((-1, 6)) / 1000
    data = data[:,0]
    print(np.std(data) / np.mean(data))
    return data.mean()

def get_computing(app, graph):
    data = pd.read_csv("result/" + graph + "-" + app + "-t.txt", header=None).to_numpy()
    data = data.reshape((-1, 6)) / 1000
    data = data[:,2]
    print(np.std(data) / np.mean(data))
    return data.mean()

def get_communication(app, graph):
    data = pd.read_csv("result/" + graph + "-" + app + "-t.txt", header=None).to_numpy()
    data = data.reshape((-1, 6)) / 1000
    data = data[:,1] + data[:,3]
    print(np.std(data) / np.mean(data))
    return data.mean()

def subway_data(app, graph):
    data = pd.read_csv("subway-result/" + graph + "-" + app + "-t.txt", header=None).to_numpy() / 1e9
    data = data.reshape((-1,3))
    print(data[:,0].std() / data[:, 0].mean(), data[:, 0].std() / data[:, 1].mean(), data.sum(axis=1).std() / data.sum(axis=1).mean())
    return data[:, 0].mean(), data[:, 1].mean(), data.sum(axis=1).mean()

apps = ["pagerank", "ppr"]
datasets = ["friendster", "uk-union"]

subway = [[subway_data(app, graph) for graph in datasets ] for app in apps]
ours = [[(get_computing(app, graph), get_communication(app, graph), get_total(app, graph)) for graph in datasets ] for app in apps]

speedup = np.array(subway) / np.array(ours)

def plot(app, app_name):
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    matplotlib.rc('pdf', fonttype=42)
    fig,ax= plt.subplots()
    fig.set_size_inches(8, 5)
    matplotlib.rcParams['savefig.dpi']=10
    matplotlib.rcParams['figure.subplot.left'] = 0
    matplotlib.rcParams['figure.subplot.bottom'] = 0
    matplotlib.rcParams['figure.subplot.right'] = 1
    matplotlib.rcParams['figure.subplot.top'] = 1
    plt.rcParams['font.sans-serif']=['Times New Roman'] 
    font1abel = {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 32
    }

    fontdatalabel = {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 26
    }
    fontlegend= {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 22,
    }

    # Assign colors for each airline and the names
    Labels = ['Computing', 'Transmission', 'Total']
    Test_Type =['FS', 'UK']

    length = len(Labels)
    num_graph = len(Test_Type)
    colors = ['#FFFF00', '#8B00AD', '#000000']
    Hatch = [''] * length
    EdgeColor = ['black'] * length

    data = speedup[app]
    
    bar_width = 0.2
    for line, _ in enumerate(Labels):
        plt.bar(1.5* np.arange(num_graph) + 0.025*line + line*bar_width - bar_width/2, data[:,line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)  

    plt.xticks(1.5*np.arange(num_graph) + 0.025 / 2,Test_Type)
    plt.tick_params(labelsize=30)
    ax.set_ylim(0,100)

    plt.legend(prop=fontlegend,frameon=True,ncol=2,loc="best",columnspacing=0.6,handletextpad=0.4)
    plt.ylabel('Speedup',font1abel)
    fig.savefig('figure10_' + app_name + '.pdf', format='pdf',bbox_inches='tight')

for app, app_name in enumerate(apps):
    plot(app, app_name)
