import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator

def getdata(file):
    data = pd.read_csv(file, header=None).to_numpy() / 1000
    return np.average(data.reshape((-1,6)), axis=0)

parts=(25, 50, 100, 150)

prefix = "result/friendster-"

data = np.zeros((4, len(parts)))

print("p Base PS SS PS+SS")
for j, p in enumerate(parts):
    print(p, end=' ')

    opt = getdata(prefix + "pagerank-opt-" + str(p) + "-t.txt")
    ss = getdata(prefix + "pagerank-asyn-" + str(p) + "-t.txt")
    base = getdata(prefix + "pagerank-base-" + str(p) + "-t.txt")
    ps = getdata(prefix + "pagerank-fifo-" + str(p) + "-t.txt")


    print(base[0], ps[0], ss[0], opt[0])

    data[0][j] = base[0]
    data[1][j] = ps[0]
    data[2][j] = ss[0]
    data[3][j] = opt[0]


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

Labels = ['Basic', 'PS','SS','PS+SS']
Test_Type =[150, 100, 50,25]
Test_Type.reverse()
colors = ['#8B00AD', '#FFFF00','#00DEA6','#000000']
Hatch = ['','','','']
EdgeColor = ['black','black','black','black']
bar_width = 0.2
for line in range(len(data)):
    plt.bar(1.5*np.arange(len(data[0]))+ line*bar_width +0.033*line,data[line],hatch=Hatch[line],edgecolor=EdgeColor[line],linewidth='1',label=Labels[line],color=colors[line],width=bar_width)

plt.xticks(1.5*np.arange(len(data[0]))+1.6*bar_width + 0.03,Test_Type)
plt.tick_params(labelsize=26)
ax.set_ylim(0,65)

plt.legend(prop=fontlegend,frameon=True,ncol=4,loc="upper left",columnspacing=0.6,handletextpad=0.4)

plt.ylabel('Total running time (s)',font1abel)
plt.xlabel('Number of partitions cached in GPU memory',font1abel)
fig.savefig('figure13.pdf',format='pdf',bbox_inches='tight')
