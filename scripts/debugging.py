"""
Jai Shri Ram
"""
import gym 
import gym_cruise_ctrl
import numpy as np
from mpc import MPCLinear
import os
import matplotlib.pyplot as plt
from plotting_utils import PlotTrainResults, PlotTestResults

"""
### Script inputs 
"""
env_version = 'cruise-ctrl-v0'
train = True
noisy_depth = False

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
T = 10
Q = np.diag([1, 1])
R = np.array([1])
modelMPC = MPCLinear(A, B, G, Q, R, T, Ts)
# x0 = [10, 2]
xref = [5.5, 0]

"""
### Run simulation
"""
### Environment state: [rel_dis (m), rel_vel (m/s), ego_vel (m/s), net_del (s)]
x0      = [10.0, 0.0]
ego_vel = 0.0
plot_test_results = PlotTestResults()

rel_dis = [x0[0]]
t = 0
while True:
    
    u, x, md = modelMPC.action(x0, xref, ego_vel)
    # print(x[0])
    # print(u)
    try:
        action = np.array([u[0,0]])
    except:
        action = np.array([0])
        print('exception')
    
    acc = 1.5*action.item()
    x0[0] = x0[0] + x0[1]*Ts - 1/2*(Ts**2)*acc
    x0[1] = x0[1] - acc*Ts
    ego_vel = ego_vel + acc*Ts
    rel_dis.append(x0[0])
  
    if t>100:
        break
    t += 1

"""
### Generate Plots
"""
fig, ax = plt.subplots()
plt.plot(rel_dis)
plt.show()