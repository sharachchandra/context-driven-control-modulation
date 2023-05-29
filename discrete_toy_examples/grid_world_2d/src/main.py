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

env = gym.make("gridworld-v2")
env.activate_shield(False, 'gym_gridworld/gym_gridworld/envs/state_action_values.npy',0.9999)
train = False
num_episodes = 100000
num_episodes_test = 10000

modelsPath = os.path.join(os.path.dirname(sys.argv[0]), 'models')
plotsPath = os.path.join(os.path.dirname(sys.argv[0]), 'plots')

if train == True:
    Q, stats = q_learning(env, num_episodes, discount_factor = 1, alpha=0.1, epsilon=0.1)
    # Working epsilon is 0.2

    Q1 = dict(Q)
    filehandler = open(os.path.join(modelsPath, 'temp.pkl'), "wb")
    pickle.dump(Q1, filehandler)

    fig1, fig2, fig3, fig4 = plotting.plot_episode_stats(stats, smoothing_window = 100,
                             noshow=True)
    fig1.savefig(os.path.join(plotsPath, 'img1.png'))
    fig2.savefig(os.path.join(plotsPath, 'img2.png'))
    fig3.savefig(os.path.join(plotsPath, 'img3.png'))
    fig4.savefig(os.path.join(plotsPath, 'img4.png'))
else:
    filehandler = open(os.path.join(modelsPath, 'Q_td2_ns.pkl'), 'rb') 
    Q = pickle.load(filehandler)
    Q = collections.defaultdict(lambda: np.zeros(env.action_space.n), Q)


stats = q_learning_test_pm(env, Q, epsilon = 0.0, num_episodes=num_episodes_test)
fig5, fig6 = plotting.plot_test_stats_pm(stats, noshow=True)
fig5.savefig(os.path.join(plotsPath, 'img5.png'))
fig6.savefig(os.path.join(plotsPath, 'img6.png'))
