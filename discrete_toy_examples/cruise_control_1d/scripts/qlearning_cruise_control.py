import gym
import gym_discrete_cruise_control

import sys 
sys.path.append("../frameworks")
from qlearning import q_learning, q_learning_test_cc

env_name = "discreteCruiseControl-v0"
env = gym.make(env_name)
env.reset()


