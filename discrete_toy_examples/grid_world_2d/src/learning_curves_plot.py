import numpy as np 
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns 
import pandas as pd

# SETTING GLOBAL PLOTTING PARAMETERS
sns.set_theme(style="darkgrid", palette="deep", color_codes='true')
FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
TICK_LABEL_SIZE = 14
plt.rc('text', usetex=False)
params = {'legend.fontsize'     : LEGEND_FONT_SIZE,
          'legend.title_fontsize': LEGEND_FONT_SIZE,
          'axes.labelsize'      : FONT_SIZE,
          'axes.titlesize'      : FONT_SIZE,
          'xtick.labelsize'     : TICK_LABEL_SIZE,
          'ytick.labelsize'     : TICK_LABEL_SIZE,
          'figure.autolayout'   : True,
          'axes.labelweight'    : 'bold',
          'font.weight'         : 'normal'
         }
plt.rcParams.update(params)

# LOADING DATA
data1 = pd.read_csv('src/paper/data/learning_curve_wo_shield_cd_0.csv').to_numpy().squeeze()
data2 = pd.read_csv('src/paper/data/learning_curve_w_shield_cd_0.csv').to_numpy().squeeze()
data3 = pd.read_csv('src/paper/data/learning_curve_wo_shield_cd_1.csv').to_numpy().squeeze()
data4 = pd.read_csv('src/paper/data/learning_curve_w_shield_cd_1.csv').to_numpy().squeeze()
data5 = pd.read_csv('src/paper/data/learning_curve_wo_shield_cd_2.csv').to_numpy().squeeze()
data6 = pd.read_csv('src/paper/data/learning_curve_w_shield_cd_2.csv').to_numpy().squeeze()

data7 = pd.read_csv('src/paper/data/min_dist_0_0.csv').to_numpy().squeeze()
data8 = pd.read_csv('src/paper/data/min_dist_0_95.csv').to_numpy().squeeze()
data9 = pd.read_csv('src/paper/data/min_dist_1_0.csv').to_numpy().squeeze()
data10 = pd.read_csv('src/paper/data/min_dist_1_95.csv').to_numpy().squeeze()
data11 = pd.read_csv('src/paper/data/min_dist_2_0.csv').to_numpy().squeeze()
data12 = pd.read_csv('src/paper/data/min_dist_2_95.csv').to_numpy().squeeze()

colors_list = sns.color_palette("deep")
#print(colors_list)

fig, ax = plt.subplots(2,3, figsize=(12,7.5))
ax[0,0].plot(data7)
ax[0,0].plot(data8)
ax[0,0].plot([5]*10000, color='black', linestyle='--')
ax[0,0].set_title('Car-following with delay 0')
ax[0,0].set_xlim(0,200)
ax[0,0].set_ylim(-2, 50)
ax[0,0].set_ylabel('Min distance (m)')

ax[0,1].plot(data9)
ax[0,1].plot(data10)
ax[0,1].plot([5]*10000, color='black', linestyle='--')
ax[0,1].set_title('Car following with delay 1')
ax[0,1].set_xlim(0,200)
ax[0,1].set_ylim(-2, 50)
ax[0,1].set_yticklabels([])

ax[0,2].plot(data11)
ax[0,2].plot(data12)
ax[0,2].set_title('Car following with delay 2')
ax[0,2].plot([5]*10000, color='black', linestyle='--')
ax[0,2].set_xlim(0,200)
ax[0,2].set_ylim(-2, 50)
ax[0,2].set_yticklabels([])

h1, = ax[1,0].plot(data1, label='Unshielded')
h2, = ax[1,0].plot(data2, label='Shielded')
ax[1,0].set_ylabel('Episode return (smoothed)')
ax[1,0].set_title('Grid world with delay 0')

ax[1,1].plot(data3)
ax[1,1].plot(data4)
ax[1,1].set_xlabel('Episodes')
ax[1,1].set_yticklabels([])
ax[1,1].set_title('Grid world with delay 1')

ax[1,2].plot(data5)
ax[1,2].plot(data6)
ax[1,2].set_yticklabels([])
ax[1,2].set_title('Grid world with delay 2')
plt.tight_layout()
# plt.show()
plt.savefig('src/paper/learning_curves.pdf',dpi=200)

figLegend = pylab.figure(figsize = (9,0.3))
pylab.figlegend(handles=[h1,h2], loc = 'upper center', ncol = 3,
                fontsize=LEGEND_FONT_SIZE, borderaxespad=0, frameon=False)
figLegend.savefig('src/paper/legends.pdf',format='pdf',dpi=200)

# ax.imshow(threshold_vis_arr, extent=[-11,11,26,-1],aspect='auto')
# ax.set_ylabel('Relative Distance (m)')
# ax.set_xlabel('Relative Velocity (m/s)')

# sdp = sns.color_palette("deep")
# p1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[4])
# a1 = mpatches.Patch(edgecolor = 'none', facecolor= sdp[7])
# ax.legend(handles=[p1,a1,a1,a1], labels=['\n','\n','  Random\n  Delay: 3 (max)','  Constant\n  Delay: 3'], 
#           loc = 'center', ncol = 2, bbox_to_anchor=(0.5,1.25),
#           handletextpad=0.0, handlelength=2.0, columnspacing=-0.0,
#           frameon=True, title='Shield Type',
#           title_fontproperties={'weight':'bold'}, alignment='right')

# plt.tight_layout()
