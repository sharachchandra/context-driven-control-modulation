import gym
import gym_gridworld
import gym_pacman
import sys
import os
import pickle
import numpy as np
import collections

import gym_gridworld.envs.gridworld as gw

from algos.q_learning import q_learning, q_learning_test_pm
import utils.plotting as plotting

# This main file is for the grid-world scenario where the controller has no idea a time-delay exists
env = gym.make("gridworld-v3")
network_delay = 2
shield_loc = 'gym_gridworld/gym_gridworld/envs/shields/state_action_values_'+str(network_delay)+'_td.npy'
active_shield = True
threshold = 0.99
# csv_file_name = 'src/paper/pmax_values_cd_' + str(network_delay) + '_delta_' + str(threshold)+'.csv'
csv_file_name = 'temp.csv'
controller_name = 'temp.pkl'

env.activate_shield(active_shield, shield_loc, threshold)
env.set_network_delay(network_delay)
env.set_pmax_csv_file_name(csv_file_name)

train = True
num_episodes = 10000
num_episodes_test = 1000

modelsPath = os.path.join(os.path.dirname(sys.argv[0]), 'models')
plotsPath = os.path.join(os.path.dirname(sys.argv[0]), 'plots')

if train == True:
    Q, stats = q_learning(env, num_episodes, discount_factor = 1, alpha=0.1, epsilon=0.1)
    # Working epsilon is 0.2

    Q1 = dict(Q)
    filehandler = open(os.path.join(modelsPath, 'temp'), "wb")
    pickle.dump(Q1, filehandler)

    fig1, fig2, fig3, fig4 = plotting.plot_episode_stats(stats, smoothing_window = 100,
                             noshow=True)
    fig1.savefig(os.path.join(plotsPath, 'img1.png'))
    fig2.savefig(os.path.join(plotsPath, 'img2.png'))
    fig3.savefig(os.path.join(plotsPath, 'img3.png'))
    fig4.savefig(os.path.join(plotsPath, 'img4.png'))
else:
    filehandler = open(os.path.join(modelsPath, controller_name), 'rb') 
    Q = pickle.load(filehandler)
    Q = collections.defaultdict(lambda: np.zeros(env.action_space.n), Q)


stats = q_learning_test_pm(env, Q, epsilon = 0.0, num_episodes=num_episodes_test)
print(stats.game_status)
# fig5, fig6 = plotting.plot_test_stats_pm(stats, noshow=True)
# fig5.savefig(os.path.join(plotsPath, 'img5.png'))
# fig6.savefig(os.path.join(plotsPath, 'img6.png'))
