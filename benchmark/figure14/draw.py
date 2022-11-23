import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

datasets = ("uk-union", "yahoo", "clueweb")
methods = "zerocopy disable opt".split(' ')
apps = ("ppr", "pagerank")


def get_data(app, method, dataset):
    file = 'result/' + dataset + '-' + method + '-' + app + "-t.txt"
    data = pd.read_csv(file, header=None).to_numpy()
    data = np.reshape(data, (5, 6))
    data = data[:,0]
    return np.mean(data) / 1000

def plot(app_name, ylim):
    data = np.zeros((len(datasets), len(methods)))
    
    for i, method in enumerate(methods):
        for j, dataset in enumerate(datasets):
            data[i][j] = get_data(app_name, 'disable', dataset) / get_data(app_name, method, dataset)
    
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    matplotlib.rc('pdf', fonttype=42)
    fig,ax= plt.subplots()
    fig.set_size_inches(10, 9)
    matplotlib.rcParams['savefig.dpi']=10
    matplotlib.rcParams['figure.subplot.left'] = 0
    matplotlib.rcParams['figure.subplot.bottom'] = 0
    matplotlib.rcParams['figure.subplot.right'] = 1
    matplotlib.rcParams['figure.subplot.top'] = 1
    plt.rcParams['font.sans-serif']=['Times New Roman'] 
    font1abel = {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 52
    }

    fontdatalabel = {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 44
    }
    fontlegend= {'family' : 'Times New Roman',
    'weight' : 'regular',
    'size' : 40,
    }

    Labels = ['All Zero Copy','All Explicit Copy','Adaptive Scheduling']
    Test_Type =['UK','YH','CW']
    colors = ['#FFFF00','#00DEA6','#000000']
    Hatch = ['','','']
    EdgeColor = ['black','black','black']
    bar_width = 0.25

    for line in range(len(data)):
        plt.bar(1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)
        if line == 0:
            y_location = 0
            for x_location in 1.5*np.arange(len(data[0]))+ line*bar_width:
                y_location += 1

    plt.xticks(1.5*np.arange(len(data[0]))+1.1*bar_width,Test_Type)
    plt.tick_params(labelsize=50)
    ax.set_ylim(0,ylim)

    y_major_locator=MultipleLocator(ylim / 4)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.legend(prop=fontlegend,frameon=True,ncol=1,loc="upper right",columnspacing=0.6,handletextpad=0.4)

    plt.ylabel('Speedup',font1abel)
    fig.savefig('figure14_' + app_name + '.pdf',format='pdf',bbox_inches='tight')

plot('pagerank', 4)
plot('ppr', 20)
