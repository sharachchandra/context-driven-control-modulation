"""
Jai Shri Ram
"""
import gym 
import gym_cruise_ctrl
import numpy as np
from mpc1 import InitializeMPC, MPCLinear
import os

from plotting_utils import PlotTrainResults, PlotTestResults

"""
### Script inputs 
"""
env_version = 'cruise-ctrl-v1'
train = False
noisy_depth = False

"""
### Initializing the environment
"""

env = gym.make(env_version, train=train, noise_required=noisy_depth)

"""
### Initialize model predictive controller
"""
modelMPC = InitializeMPC(env.delt)

"""
### Run simulation
"""
### Environment state: [rel_dis (m), rel_vel (m/s), ego_vel (m/s), net_del (s)]
obs = env.reset()
plot_test_results = PlotTestResults()

while True:
    
    u, x, md = modelMPC.action(obs)
    print(u[0,0])
    try:
        action = np.array([u[0,0]])
    except:
        action = np.array([0])
        print('exception') 
    # action = np.array([0.1])
    obs, reward, done, info = env.step(action)

    plot_test_results.store(obs, reward, info) # Gather results for plotting
  
    if done:
        break
env.close() 

"""
### Generate Plots
"""

plot_test_results.plot()