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

data = np.zeros((3,6))

print("partition-size walk reshuffle batch")
for i in range(6):
    print(32 * (2 ** i), end=' ')
    time, call = get_avg("result/kernel-opt-" + str(i) + "-")
    print(time["walk"], end=' ')
    print(time["reshuffle"], end=' ')
    print(time["batch"])

    data[0][i] = time["walk"]
    data[1][i] = time["reshuffle"]
    data[2][i] = time["batch"]

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
'size' : 20,
}

# Assign colors for each airline and the names
Labels = ['Walk Updating','Walk Reshuffling','Other','Total']
Test_Type = [32,64,128,256,512,1024]


colors = ['#8B00AD','#00DEA6','#FFFF00']
Hatch = ['','','']
EdgeColor = ['black','black','black']

bar_width = 0.40

line=0
Bottom = 0
for stack in range(3):
    plt.bar(1.5*np.arange(len(data[0])),data[stack],hatch=Hatch[stack],edgecolor=EdgeColor[stack],linewidth='1',label=Labels[stack],color=colors[stack],width=bar_width,bottom=Bottom)
    Bottom = Bottom + data[stack]

    
plt.xticks(1.5*np.arange(len(data[0])),Test_Type)
plt.tick_params(labelsize=26)
ax.set_ylim(0,10)

y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop=fontlegend,frameon=False,ncol=4,loc="upper center",columnspacing=0.6,handletextpad=0.4,bbox_to_anchor=(0,0.8,0.97,0.2))

plt.ylabel('Computation Time (s)',font1abel)
plt.xlabel('Partition size (MB)',font1abel)
fig.savefig('figure17.pdf',format='pdf',bbox_inches='tight')
