"""
Jai Shri Ram 
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from os import path
import pygame
from pygame import gfxdraw
from gym_cruise_ctrl.envs.input_generator import InputAccGenerator
from gym_cruise_ctrl.envs.noise_model import NoisyDepth, NoisyVel
from gym_cruise_ctrl.envs.network_strength import NetworkStrength
from gym_cruise_ctrl.envs.delay_handler import DelayHandler

class CruiseCtrlEnv0(gym.Env):

	def __init__(self, train=True, noise_required=False): 

		"""
		### Action Space
			The action is a scalar in the range `[-1, 1]` that multiplies the max_acc
			to give the acceleration of the ego vehicle. 
		"""
		self.action_low   = -1.0
		self.action_high  =  1.0
		self.action_space = gym.spaces.Box(low=self.action_low, high=self.action_high, shape=(1,))
		
		"""
		### Observation Space
			The observation is an ndarray of shape (4,) with each element in the range
			`[-inf, inf]`.   
			1. Relative distance-1 (noisy)
			2. Relative velocity-1 (noisy)
			3. Velocity
			4. Network delay (seconds)
		"""
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

		"""
		### Episodic Task
		"""
		self.delt = 1										# 0.1s time step
		self.max_episode_steps = int(100/self.delt)			# 1000 episode length

		"""
		### Environment Specifications   
		"""
		### Safety specifications
		self.safety_dist 				   = 5				# Required distance between the ego and front vehicle
		self.violating_safety_dist_reward  = -10*self.delt	# Reward for getting too close to the front car
		self.jerk_cost_coef				   = 0.1

		### Front vehicle specifications
		self.fv_min_vel = 10 								# 10m/s or 30mph
		self.fv_max_vel = 30 								# 30m/s or 70mph
		self.fv_max_acc = 0.5  								# 0.5m/s^2

		### Ego vehicle specifications		
		self.ego_max_vel  = 40 								# 40m/s or 90mph 
		self.ego_max_acc  =  1.5
		self.ego_max_dist = self.ego_max_vel*self.delt      # Max distance travelled in one time step
		
		### Noise specifications
		self.noise_required    = noise_required 			# Whether noise is required or not
		self.depth_noise_model = NoisyDepth() 				# Depth noise model class
		self.vel_noise_model   = NoisyVel() 				# Velocity noise model class 

		### For random seed purposes 
		self.train = train 									# Are we training or validating? For validating, we set the seed to get constant initializations

		"""
		### Visualizer Parameters
		"""
		self.screen_dim = 500
		self.screen = None

		"""
		### Network latency
		"""
		self.delay_per_level   = 1 		# In seconds
		self.max_network_level = 1		# The network strenght is an integer from 0 to max_network_level

		"""
		### Initialziation
		"""
		self.InitializeEnvironmentVariables()

	def InitializeEnvironmentVariables(self):
		self.episode_steps = 0
		self.done = False

		"""
		### Initial conditions
		"""
		### Front vehicle
		self.fv_init_pos = self.InitializeFvPos()
		self.fv_init_vel = self.InitializeFvVel()
		self.fv_state    = np.array([self.fv_init_pos, self.fv_init_vel], dtype=np.float32)
		self.fv_acc_list = InputAccGenerator(self.max_episode_steps, self.delt, self.fv_init_vel, 
											 self.fv_min_vel, self.fv_max_acc, self.fv_max_acc, 
											 mode = 'SplineProfile')

		### Ego vehicle
		self.ego_init_pos = self.InitializeEgoPos()
		self.ego_init_vel = self.InitializeEgoVel()
		self.ego_state    = np.array([self.ego_init_pos, self.ego_init_vel], dtype=np.float32) 
		self.prev_acc     = 0.0
		
		rel_pose = self.fv_state - self.ego_state

		### Network latency
		self.ns_ts = NetworkStrength(self.max_episode_steps, n_min = 1, n_max = 3, low = 0, high = self.max_network_level)
		self.nd_ts = self.ns_ts*self.delay_per_level

		### State value
		self.state = np.append(rel_pose,   self.ego_init_vel)
		self.state = np.append(self.state, self.nd_ts[0])


	def InitializeFvPos(self):
		if self.train:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m
		else:
			return max(20*np.random.randn() + 100, 10) # (100 +- 20)m

	def InitializeFvVel(self):
		if self.train:
			return 0
		else:
			return min(self.fv_max_vel, max(self.fv_min_vel + np.random.rand()*(self.fv_max_vel - self.fv_min_vel), 
											self.fv_min_vel))	# (20 +-5)m/s or 60mph

	def InitializeEgoPos(self):
		return 0

	def InitializeEgoVel(self):
		if self.train:
			return max(5*np.random.randn() + 10, 0)	# (10 +-5)m/s
		else:
			return min(self.fv_max_vel, max(self.fv_min_vel + np.random.rand()*(self.fv_max_vel - self.fv_min_vel), 
											self.fv_min_vel))


	def step(self, action):
		fv_pos  = self.fv_state[0]
		fv_vel  = self.fv_state[1]
		ego_pos = self.ego_state[0]
		ego_vel = self.ego_state[1] 

		"""
		### Front vehicle state transition
		"""
		### Include system model
		### Acceleration input to the front vehicle
		fv_acc = self.fv_acc_list[self.episode_steps]
		if self.train:
			fv_acc = 0.0
		else:
			fv_acc = fv_acc*self.fv_max_acc
		
		### Clipping acceleration to keep within velocity limits
		if fv_vel >= self.fv_max_vel:
			if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
				fv_acc = 0.0
		else:
			if fv_vel + fv_acc*self.delt >= self.fv_max_vel:
				fv_acc = (self.fv_max_vel - fv_vel)/self.delt
		
		if fv_vel <= self.fv_min_vel:
			if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
				fv_acc = 0.0
		else:
			if fv_vel + fv_acc*self.delt <= self.fv_min_vel:
				fv_acc = (self.fv_min_vel - fv_vel)/self.delt		

		### State update
		fv_pos = fv_pos + fv_vel*self.delt + 0.5*fv_acc*self.delt**2
		fv_vel = fv_vel + fv_acc*self.delt
		self.fv_state = np.array([fv_pos, fv_vel], dtype=np.float32)
		
		"""
		### Ego vehicle state transition
		"""
		### Include system model
		### Acceleration input to the ego vehicle
		# action  = np.clip(action, self.action_low, self.action_high)[0]
		ego_acc = action.item()*self.ego_max_acc

		### Clipping acceleration to keep within velocity limits
		if ego_vel >= self.ego_max_vel:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = 0.0
		else:
			if ego_vel + ego_acc*self.delt >= self.ego_max_vel:
				ego_acc = (self.ego_max_vel - ego_vel)/self.delt

		### State update
		ego_dist_trav = ego_vel*self.delt + 0.5*ego_acc*self.delt**2
		ego_pos = ego_pos + ego_dist_trav 
		ego_vel = ego_vel + ego_acc*self.delt
		self.ego_state = np.array([ego_pos, ego_vel], dtype=np.float32)

		"""
		Network latency
		"""
		nd = self.nd_ts[self.episode_steps]

		"""
		# Reward function
		"""
		### Reward for moving forward
		reward = ego_dist_trav/self.ego_max_dist
		
		### Jerk cost function
		jerk = abs(ego_acc - self.prev_acc)
		self.prev_acc = ego_acc
		# reward -= self.jerk_cost_coef*jerk

		### Reward for being too close to the front vehicle
		rel_dis = fv_pos - ego_pos
		if rel_dis < self.safety_dist:
			reward += self.violating_safety_dist_reward

		"""
		# Noise corruption
		"""
		rel_pose = self.fv_state - self.ego_state
		# rel_pose_noisy = np.array([self.depth_noise_model(rel_pose[0]), 
		# 						   self.vel_noise_model(rel_pose[1], rel_pose[0])]).flatten()
		# rel_pose_laggy = np.array([self.delay_handler_rel_dis.update(rel_pose[0], ns), 
		# 						   self.delay_handler_rel_vel.update(rel_pose[1], ns)]).flatten()

		"""
		### States
		"""
		self.state = np.append(rel_pose,   ego_vel)
		self.state = np.append(self.state, nd)
		
		"""
		### Observation
		"""
		obs = self.state.copy()

		"""
		### Environment handling 
		"""
		### Terminating the episode
		if rel_dis <= 2 or self.episode_steps >= self.max_episode_steps:
			self.done = True 

		self.episode_steps += 1

		info = {
			"fv_pos"  : fv_pos,
			"fv_vel"  : fv_vel,
			"fv_acc"  : fv_acc, 
			"ego_pos" : ego_pos,
			"ego_vel" : ego_vel,
			"ego_acc" : ego_acc,
		}

		return obs, reward, self.done, info

	def reset(self, seed=0):
		# if not self.train:
		# 	np.random.seed(seed)
		self.InitializeEnvironmentVariables()
	
		obs = self.state.copy()

		return obs 

	def render(self, mode='human'):
		
		if self.screen == None:
			pygame.init()
			pygame.display.init()

			self.screen = pygame.display.set_mode([self.screen_dim, self.screen_dim])

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
				pygame.quit()

		self.screen.fill((255, 255, 255))
		pygame.draw.circle(self.screen, (0,0,255), (self.screen_dim*9/10,self.screen_dim/2), 15)
		pygame.draw.circle(self.screen, (255,0,0), (self.screen_dim*(9/10 - 2/10*self.state[0]/self.fv_init_pos), self.screen_dim/2), 15)
		pygame.display.flip()
		pygame.time.delay(33)

	def close(self):
		pygame.display.quit()
		pygame.quit()

