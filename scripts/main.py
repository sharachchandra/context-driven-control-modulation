"""
Jai Shri Ram
"""
import gym 
import gym_cruise_ctrl
import numpy as np
from mpc import MPCLinear
import os

from plotting_utils import PlotTrainResults, PlotTestResults

"""
### Script inputs 
"""
env_version = 'cruise-ctrl-v0'
train = False
noisy_depth = False

"""
### Initializing the environment
"""

env = gym.make(env_version, train=train, noise_required=noisy_depth)

"""
### Initialize model predictive controller
"""
# xref = [5.5, 0]
# modelMPC = InitializeMPC(env.delt)

Ts = 1
A = np.array([[1, Ts],
              [0,  1]])
B = 1.5*np.array([[-0.5*Ts*Ts],
              [-Ts]])
G = np.array([[0.5*Ts*Ts],
              [Ts]])
T = 20
Q = np.diag([10, 10])
R = np.array([1])
modelMPC = MPCLinear(A, B, G, Q, R, T, Ts)
# x0 = [10, 2]

"""
### Run simulation
"""
### Environment state: [rel_dis (m), rel_vel (m/s), ego_vel (m/s), net_del (s)]
obs = env.reset()
plot_test_results = PlotTestResults()

while True:
    
    u, x, md = modelMPC.action(obs)
    # print(x[0])
    # print(u)
    try:
        action = np.array([u[0,0]])
    except:
        action = np.array([0])
        print('exception') 
    obs, reward, done, info = env.step(action)

    plot_test_results.store(obs, reward, info) # Gather results for plotting
  
    if done:
        break
env.close() 

"""
### Generate Plots
"""

plot_test_results.plot()