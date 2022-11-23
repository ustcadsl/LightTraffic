import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

def get_avg(file_name):
    iteration = 5
    df = []
    for i in range(iteration):
        df.append(pd.read_csv(file_name + str(i) +".txt", thousands=r',', sep=" ", index_col=0))

    time = {"walk": 0, "reshuffle": 0, "batch": 0}
    call = {"walk": 0, "reshuffle": 0, "batch": 0}

    for i in range(iteration):
        for ax in df[i].axes[0]:
            if ax[:4] == "rand":
                time["walk"] += df[i].at[ax, "time"]
                call["walk"] += df[i].at[ax, "calls"]
            elif ax[:4] == "inse":
                time["reshuffle"] += df[i].at[ax, "time"]
                call["reshuffle"] += df[i].at[ax, "calls"]
            elif ax[:4] == "page":
                time["batch"] += df[i].at[ax, "time"]
                call["batch"] += df[i].at[ax, "calls"]

    for k in time.keys():
        time[k] /= iteration * (1000 ** 3)
        call[k] /= iteration * (1000 ** 3)

    return time, call

data = np.zeros((2,6))

print("partition-size 3-reshuffle-method-time")
for i in range(6):
    print(32 * (2 ** i), end=' ')
    time, _ = get_avg("result/kernel-baseline1-" + str(i) + "-")
    print(time["reshuffle"], end=' ')
    data[0][i] = time["reshuffle"]

    time, _ = get_avg("result/kernel-opt-" + str(i) + "-")
    print(time["reshuffle"])
    data[1][i] = time["reshuffle"]

plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
matplotlib.rc('pdf', fonttype=42)
fig,ax= plt.subplots()
fig.set_size_inches(9, 5)
matplotlib.rcParams['savefig.dpi']=10
matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1
plt.rcParams['font.sans-serif']=['Times New Roman'] 
font1abel = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 26
}

fontdatalabel = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 20
}
fontlegend= {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 24,
}

# Assign colors for each airline and the names
Labels = ['Direct Write','Two-level Caching', ]
Test_Type =[32, 64, 128, 256, 512, 1024]


colors = ['#FFFF00','#00DEA6','#000000']
Hatch = ['','','']
EdgeColor = ['black','black','black']

bar_width = 0.35
for line in range(len(data)):
    plt.bar(1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)
plt.xticks(1.5*np.arange(len(data[0]))+0.55*bar_width,Test_Type)
plt.tick_params(labelsize=26)
ax.set_ylim(0,5)

y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop=fontlegend,frameon=True,ncol=2,loc="upper right",columnspacing=0.6,handletextpad=0.4)

plt.ylabel('Reshuffling time (s)',font1abel)
plt.xlabel('Partition size (MB)',font1abel)
fig.savefig('figure12.pdf',format='pdf',bbox_inches='tight')
