import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

print(os.getcwd())
sns.set_style('darkgrid')
df = pd.DataFrame({ 'Threshold':[   'No Shield', '1.00', '0.999', 
                                    'No Shield', '1.00', '0.999', 
                                    'No Shield', '1.00', '0.999'],
                    'Percentage':[  73, 25, 71, 
                                    13, 75, 29, 
                                    12, 0, 0.02],
                    'Status':[  'Win','Win','Win',
                                'Tie','Tie','Tie',
                                'Loss','Loss','Loss']})
print(df)
sns_plot = sns.barplot(data=df, x='Threshold', y='Percentage', hue='Status')
df.plot(kind='bar', stacked=True)
sns_plot.set(ylim=(0,100))
sns_plot.figure.savefig('src/paper/time_delay_2_percentages.png')

# plotsPath = os.path.join(os.path.dirname(sys.argv[0]), 'plots')
# dataPath  = os.path.join(os.path.dirname(sys.argv[0]), 'data')
# wo_shield = pd.read_csv(os.path.join(dataPath, 'learning_curve_wo_shield.csv'), header=None)
# w_shield  = pd.read_csv(os.path.join(dataPath, 'learning_curve_w_shield.csv'), header=None)

# fig, axes = plt.subplots(1,1)
# axes.plot(wo_shield, label = 'No shielding',color='red')
# axes.plot(w_shield,  label = 'With shielding',color='blue')
# # axes.plot(*zip(*stats.adversary), label = 'adversary path', color='red')
# # axes.plot(*stats.robot[-1], marker='o',color='blue')
# # axes.plot(*stats.adversary[-1], marker='o',color='red')
# plt.xlabel("Episodes")
# plt.ylabel("Episode Reward (Smoothed)")
# plt.title("Episode Reward over Time")
# # axes.set_xlim([-1, 10])
# # axes.set_ylim([-1, 10])
# axes.grid()
# axes.legend()
# fig.savefig(os.path.join(plotsPath, 'learning_curves.png'))