import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

systems = ['thunderrw', 'flashmob', 'lighttraffic', 'pcie3', 'pcie4']
throughput = {}
err = {}
for sys in systems:
    data = pd.read_csv(sys + '_throughput.txt', header=None).to_numpy()
    data = data.reshape((-1,4,5)) / 1e9
    err[sys] = np.std(data, axis=2)
    throughput[sys] = np.mean(data, axis=2)
    
throughput['flashmob'] = np.append(throughput['flashmob'], np.zeros((1,4)), axis=0)
err['flashmob'] = np.append(err['flashmob'], np.zeros((1,4)), axis=0)

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
    'size' : 24,
    }

    # Assign colors for each airline and the names
    Labels = ['ThunderRW', 'FlashMob', 'LT (PCIe 3.0 3090)', 'LT (PCIe 3.0 A100)', 'LT (PCIe 4.0 A100)']
    Test_Type =['FS', 'UK', 'YH', 'CW']

    length = len(throughput)
    num_graph = 4
    colors = ['#FFFF00','#00DEA6', '#E0E0E0' ,'#A9A9A9','#000000']
    Hatch = [''] * length
    EdgeColor = ['black'] * length

    bar_width = 0.2
    for line, sys in enumerate(systems):
        plt.bar(1.5* np.arange(num_graph) + line*bar_width +0.033*line - bar_width/2 - 0.033, throughput[sys][app],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)  
        plt.errorbar(1.5* np.arange(num_graph) + line*bar_width +0.033*line - bar_width/2 - 0.033, throughput[sys][app], yerr=err[sys][app]*1.96, alpha=0.5, ecolor='red', fmt='_', capsize=5, linewidth=3)

    plt.xticks(1.5*np.arange(num_graph)+1.65*bar_width,Test_Type)
    plt.tick_params(labelsize=30)
    ax.set_ylim(0,5)

    y_major_locator=MultipleLocator(2)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.legend(prop=fontlegend,frameon=True,ncol=2,loc="best",columnspacing=0.6,handletextpad=0.4)

    plt.ylabel('Throughput (Gstep/s)',font1abel)
    fig.savefig('figure9_' + app_name + '.pdf', format='pdf',bbox_inches='tight')

for app, app_name in enumerate(["deepwalk", "pagerank", "ppr"]):
    plot(app, app_name)
