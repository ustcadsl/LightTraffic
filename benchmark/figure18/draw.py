import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

D = np.array([((2 ** i), (2 ** i) * 1.5) for i in range(-7, 2)]).flatten()

fs_walk = np.array(D * 15041590484 / 8, dtype='int64')[:-1]
uk_walk = np.array(D * 38359720188 / 8, dtype='int64')[:-1]
cw_walk = np.array(D * 75970033200 / 8, dtype='int64')[:-3]

iteration = 5
def get_running_time(dataset):
    data = pd.read_csv(dataset + ".txt", header=None).to_numpy()
    data = data / 1000
    data = data.reshape((-1, iteration, 6))
    data = data[:,:,0]
    data = np.average(data, axis=1)
    return data

def theoretical_tp(D):
    return (1.5 * (2 ** 30)) / ( 1 + 1 / max(1/32, D))

throughput = [0] * 4
density = [0] * 4

throughput[0] = np.array([theoretical_tp(0.01 * i) for i in range(0, 200)]) / 1e9
density[0] = np.arange(200) * 0.01

throughput[1] = fs_walk * 10 / get_running_time('fs') / 1e9
density[1] = D[:-1]
throughput[2] = cw_walk * 10 / get_running_time('cw') / 1e9
density[2] = D[:-3]
throughput[3] = uk_walk * 10 / get_running_time('uk') / 1e9
density[3] = D[:-1]

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
'size' : 30
}
fontdatalabel = {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 20
}
fontlegend= {'family' : 'Times New Roman',
'weight' : 'regular',
'size' : 28,
}

Labels = ["Theory", "FS", "CW", 'UK']
colors = ['#F057FF','#00DEA6','#FFFF00', '#00FFFF']
Hatch = ['','','']
EdgeColor = ['black'] * len(Labels)
marks=['s','o','d','>','P','*']
style=['-','-','-','-','-.',':']

plt.plot(density[0], throughput[0], label=Labels[0], linewidth=2, linestyle=style[0])

for line in range(1, len(Labels)):
    plt.plot(density[line], throughput[line], label=Labels[line],marker=marks[line],linewidth=4,markersize=6,linestyle=style[line])

plt.tick_params(labelsize=26)
plt.legend(prop=fontlegend,frameon=False,ncol=2,loc="best",columnspacing=0.6,handletextpad=0.4,markerscale=1.5)
plt.xlabel('Walk density',font1abel)
plt.ylabel('Throughput (Gstep/s)',font1abel)
fig.savefig('figure18.pdf',format='pdf',bbox_inches='tight')
