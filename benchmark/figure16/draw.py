import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

def read_data(filename):
    data = pd.read_csv(filename, header=None)
    data = data.to_numpy()
    data = data.reshape((5, 16, 6))
    data = np.mean(data, axis=0)
    data = data.T
    order = [0,1,3,4,5,2]
    data = data[order]
    return data

multirun = read_data("mem-size-data.txt")
onerun = read_data("../figure15/mem-size-data.txt")

speedup = multirun[0,:] / onerun[0,:]
speedup = np.reshape(speedup, (4,4))
speedup = speedup[:3,:3]
speedup = speedup.T
data = speedup[::-1,:]


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
'size' : 24
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
Labels = ['400M × 2 runs','200M × 4 runs','100M × 8 runs']
Test_Type =[25, 50, 100]

colors = ['#FFFF00','#00DEA6','#8B00AD']
Hatch = ['','','']
EdgeColor = ['black','black','black']

bar_width = 0.3
for line in range(len(data)):
    plt.bar(1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)

plt.xticks(1.5*np.arange(len(data[0]))+1.1*bar_width,Test_Type)
plt.tick_params(labelsize=22)
ax.set_ylim(0,5.5)

y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)

plt.legend(prop=fontlegend,frameon=True,ncol=1,loc="best",columnspacing=0.6,handletextpad=0.4)

plt.ylabel('Slowdown using multiple rounds',font1abel)
plt.xlabel('Number of graph partitions cached in GPU memory',font1abel)
ax.set_xlim(-0.5,4.2)
plt.plot(np.array(range(-1,12))/2,[1 for i in range(-1,12)],linestyle = 'dotted', color='black')
fig.savefig('figure16.pdf',format='pdf',bbox_inches='tight')
