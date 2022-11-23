import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

data = pd.read_csv("mem-size-data.txt", header=None)

data = data.to_numpy()
data = data.reshape((5, 16, 6))

data = np.mean(data, axis=0)

data = data.T
order = [0,1,3,4,5,2]
data = data[order]
data=data/1000.0

group = [0] * 4
for i in range(4):
    group[i] = data[:,i * 4: (i + 1) * 4]

plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
matplotlib.rc('pdf', fonttype=42)
fig,ax= plt.subplots()
fig.set_size_inches(24, 6)
matplotlib.rcParams['savefig.dpi']=10
matplotlib.rcParams['figure.subplot.left'] = 0
matplotlib.rcParams['figure.subplot.bottom'] = 0
matplotlib.rcParams['figure.subplot.right'] = 1
matplotlib.rcParams['figure.subplot.top'] = 1
plt.rcParams['font.sans-serif']=['Times New Roman'] 
font1abel = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 28
}
fontdatalabel = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 20
},
x_label = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 10
}
fontlegend= {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 28,
}

# Assign colors for each airline and the names
Labels = ['Total Time','Graph Loading','Zero Copy','Walk Loading','Walk Eviction','Walk Computing' ]
Test_Type =['1, 1', '2, 1', '4, 1', '8, 1',\
    '1, 2', '2, 2', '4, 2', '8, 2',\
    '1, 4', '2, 4', '4, 4', '8, 4',\
    '1, 8', '2, 8', '4, 8', '8, 8']

#E9967A  #FFA07A #FA8072
colors = ['#FFFF00','#00DEA6','#ffffff','#ffffff','#8B00AD','#000000']
#colors = ['#FFFF00','#00DEA6','#FFB6C1','#2166F5','#F057FF','#000000']
Hatch = ['','','\\\\','xx','','']
#Hatch = ['','','','','','']
EdgeColor = ['black','black','#FF3030','#2166F5','black','black']
colors.reverse()
Hatch.reverse()
EdgeColor.reverse()

bar_width = 0.4

for g_id in range(len(group)):
    data = group[g_id]
    
    line = 1
    Bottom = 0
    for stack in range(1,6):
        plt.bar(1.5 * 4.5 * g_id + 1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[stack],hatch=Hatch[stack],edgecolor=EdgeColor[stack],linewidth='1', color=colors[stack],width=bar_width,bottom=Bottom, label=Labels[stack] if g_id == 0 else None)
        Bottom = Bottom + data[stack]

    line = 0
    plt.bar(1.5 * 4.5 * g_id + 1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1', color=colors[line],width=bar_width, label=Labels[line] if g_id == 0 else None)
    plt.bar(1.5 * 4.5 * g_id + 1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line, Bottom - data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1', color='white', width=bar_width, bottom=data[line], label="Time hidden by pipeline" if g_id == 0 else None)

        
        
plt.tick_params(labelsize=23)
x_label  = []
for i in range(4):
    x_label += list(1.5*i*4.5+1.5*np.arange(4)+0.55*bar_width)
plt.xticks(x_label,Test_Type)

ax.set_ylim(0,22)

y_major_locator=MultipleLocator(5)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop=fontlegend,frameon=False,ncol=3,loc="best",columnspacing=0.6,handletextpad=0.4,bbox_to_anchor=(0,0.8,1,0.2))

plt.ylabel('Time (s)',font1abel)
plt.xlabel('Number of walks (in 100 million), number of graph partitions (× 25) cached in GPU memory',font1abel)
fig.savefig('figure15.pdf',format='pdf',bbox_inches='tight')
