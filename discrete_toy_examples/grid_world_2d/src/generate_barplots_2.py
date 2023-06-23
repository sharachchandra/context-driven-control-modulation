import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sns.set_theme()
sns.set_color_codes(palette="pastel")
FONT_SIZE = 16


plt.rc('text', usetex=False)
plt.rcParams['text.latex.preamble'] = [r'\boldmath'] # makes math symbols only bold

LEGEND_FONT_SIZE = 12
XTICK_LABEL_SIZE = 12
YTICK_LABEL_SIZE = 12

plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["font.weight"] = "bold"
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import matplotlib.pylab as pylab
params = {'legend.fontsize': LEGEND_FONT_SIZE,
          'legend.title_fontsize': LEGEND_FONT_SIZE,
         'axes.labelsize': FONT_SIZE,
         'axes.titlesize': FONT_SIZE,
         'xtick.labelsize': XTICK_LABEL_SIZE,
         'ytick.labelsize': YTICK_LABEL_SIZE,
         'figure.autolayout': True}
pylab.rcParams.update(params)
plt.rcParams["axes.labelweight"] = "bold"

index = ["Constant \n Delay: 0", 
         "Constant \n Delay: 1", 
         "Constant \n Delay: 2", 
         "Constant \n Delay: 3",
         "Random \n Delay: 3(max)"]
stacks = ["Loss", "Win", "Tie"]

# create fake dataframes
df1 = pd.DataFrame([[0,100,0],
                    [15,85,0],
                    [14,86,0],
                    [14,86,0],
                    [1,98,1]],
                   index=index,
                   columns=stacks)
df2 = pd.DataFrame([[0,100,0],
                    [5,89,6],
                    [9,83,8],
                    [11,71,18],
                    [0,99,1]],
                   index=index,
                   columns=stacks)
df3 = pd.DataFrame([[0,100,0],
                    [0,90,10],
                    [3,73,24],
                    [8,65,27],
                    [0,99,1]],
                   index=index,
                   columns=stacks)
df4 = pd.DataFrame([[0,100,0],
                    [0,88,12],
                    [0,68,32],
                    [0,55,45],
                    [0,97,3]],
                   index=index,
                   columns=stacks)
df5 = pd.DataFrame([[0,100,0],
                    [0,79,21],
                    [0,74,26],
                    [0,66,34],
                    [0,75,25]],
                   index=index,
                   columns=stacks)

df1["Threshold"] = "0.0"
df2["Threshold"] = "0.9"
df3["Threshold"] = "0.95"
df4["Threshold"] = "0.999"
df5["Threshold"] = "1.0"

dfall = pd.concat([pd.melt(i.reset_index(names=["Time Delay"]), 
                   id_vars=["Threshold", "Time Delay"],
                   var_name="Game status",
                   value_name="value") # transform in tidy format each df
                   for i in [df1, df2, df3, df4, df5]],
                   ignore_index=True)

dfall.set_index(["Threshold", "Time Delay", "Game status"], inplace=True)
dfall["Outcome Percentage"] = dfall.groupby(level=["Threshold", "Time Delay"]).cumsum()
dfall.reset_index(inplace=True) 
print(dfall)

# Using seaborn's deep color palette's red blue and green
deep_palette = sns.color_palette("deep").as_hex()
# sns.palplot(deep_palette)
# plt.show()

green_sns  = deep_palette[2]
blue_sns   = deep_palette[0]
orange_sns = deep_palette[1]

colors = [orange_sns,
          green_sns,
          blue_sns]
# colors = [sns.xkcd_rgb["medium green"], 
#         sns.xkcd_rgb["denim blue"], 
#         sns.xkcd_rgb["pale red"], 
#         sns.xkcd_rgb["amber"], 
#         sns.xkcd_rgb["dusty purple"], 
#         sns.xkcd_rgb["greyish"]]

hatches = ['**','','\\\\']

fig, ax = plt.subplots(figsize=(7.4,3.8))
groups = dfall.groupby("Game status")
for i, k in enumerate(stacks):
    sns.barplot(ax=ax,data=groups.get_group(k),
                     x="Time Delay",
                     y="Outcome Percentage",
                     hue="Threshold",
                     palette=[colors[i]]*5,
                     zorder=-i, # so first bars stay on top
                #      hatch=hatches[i],
                     edgecolor="k")
ax.legend_.remove() # remove the redundant legends
ax.set_xlabel('')

lp = mpatches.Patch(edgecolor = orange_sns, facecolor= orange_sns, label='Loss')
wp = mpatches.Patch(edgecolor = green_sns, facecolor= green_sns, label='Win')
bp = mpatches.Patch(edgecolor = blue_sns, facecolor= blue_sns, label='Tie')

# t1 = mpatches.Patch(color=colors[0], label='0')
# t2 = mpatches.Patch(color=colors[1], label='0.90')
# t3 = mpatches.Patch(color=colors[2], label='0.95')
# t4 = mpatches.Patch(color=colors[3], label='0.99')
# t5 = mpatches.Patch(color=colors[4], label='1')

# legend1 = plt.legend(title="Threshold",handles=[t1,t2,t3,t4,t5], loc="upper left", bbox_to_anchor=(1, 1))

legend2 = plt.legend(handles=[wp,lp,bp], loc="lower left", ncol = 1, bbox_to_anchor=(1.0, 0.5),frameon=False,
                     title='Game Status',title_fontproperties={'weight':'bold'}, alignment='left')
ax.xaxis.set_tick_params(bottom=False, labelbottom = True)
# ax.set_title("Constant Time Delay")
ax2 = ax.twiny()
dt = 0.03
tick_loc = np.array(5*[0.1] + 5*[0.3] + 5*[0.5] + 5*[0.7] + 5*[0.9]) + \
           np.array(5*[-2*dt, -dt, 0.0, dt, 2*dt])

ax2.set_xticks(tick_loc)
ax2.set_xticklabels(5*['0','0.90','0.95','0.99','1'])
ax2.xaxis.set_tick_params(labeltop = True, 
                          top = True, 
                          labelbottom = False, 
                          bottom=False,
                          rotation=90)

ax2.grid(False)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off                      
# ax.add_artist(legend1)
# ax.add_artist(legend2)
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.title('Threshold', fontsize=LEGEND_FONT_SIZE, weight='bold')
plt.tight_layout()
plt.savefig('src/paper/gw_emp_RAL.pdf')