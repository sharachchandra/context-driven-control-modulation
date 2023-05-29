import gym
import numpy as np
import math
import os
import sys

class Gridworld2(gym.Env):
    def __init__(self) -> None:
        super().__init__()
    
        self.n_action = 5
        self.action_space = gym.spaces.Discrete(self.n_action)
        
        # Network delay
        self.time_delay = 2 # 2 time steps

        self.n = 10
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(5),
                                                   gym.spaces.Discrete(5)))
        # (row, column)
        self.max_episode_steps = 4*self.n
        self.goal = (self.n-1,self.n-1)
    
        self.shield_active = False
        self.probDict = None
        self.prob_threshold = None

        # self.shieldDict = np.load('shield.npy', allow_pickle = True).item()
        # self.probDict = np.load('state_action_values.npy', allow_pickle = True).item()
        self.shield = self.Shield()
        self.shieldDict = np.load('gym_gridworld/gym_gridworld/envs/shield.npy', allow_pickle = True).item()
      
    class Shield():
        def __init__(self) -> None:
            pass
        
        def get_shielded_actions(self, state):
            rob_state = np.array(state[0:2])
            obs_state = np.array(state[2:4])
            rel_state = obs_state - rob_state

            if   np.array_equal(rel_state, [0, 2]):
                return (0, 2, 3, 4)
            elif np.array_equal(rel_state, [2, 0]):
                return (0, 1, 3, 4)
            elif np.array_equal(rel_state, [0, -2]):
                return (0, 1, 2, 4)
            elif np.array_equal(rel_state, [-2, 0]):
                return (0, 1, 2, 3)

            elif np.array_equal(rel_state, [0, 1]):
                return (2, 3, 4)
            elif np.array_equal(rel_state, [1, 0]):
                return (1, 3, 4)
            elif np.array_equal(rel_state, [0, -1]):
                return (1, 2, 4)
            elif np.array_equal(rel_state, [-1, 0]):
                return (1, 2, 3)

            elif np.array_equal(rel_state, [1, 1]):
                return (0, 3, 4)
            elif np.array_equal(rel_state, [1, -1]):
                return (0, 1, 4)
            elif np.array_equal(rel_state, [-1, -1]):
                return (0, 1, 2)
            elif np.array_equal(rel_state, [-1, 1]):
                return (0, 2, 3)

            elif np.array_equal(rel_state, [0, 0]):
                return ()
            
            else:
                return (0, 1, 2, 3, 4)
    
    def activate_shield(self, val, path, threshold):
        self.shield_active = val
        self.probDict = np.load(path, allow_pickle = True).item()
        self.prob_threshold = threshold
    
    def get_shielded_action(self, action, state):
        if self.time_delay == 0:
            shielded_actions = self.shield.get_shielded_actions(state)        
        else:
            n = int((4**self.time_delay - 1)/3)
            rob_state = state[0:2]
            all_states = []
            all_states.append(state)
            for _ in range(n):
                pop_state = all_states.pop(0)
                
                for i in range(1, 5):
                    next_state_adv = self.trans(pop_state[2:4], i)
                    next_state = (*rob_state, *next_state_adv)
                    all_states.append(next_state)
            
            shielded_actions_list = []
            shielded_actions = self.shield.get_shielded_actions(all_states[0])
            shielded_actions_list.append(shielded_actions)
            for i in range(1, len(all_states)):
                allowable_actions = self.shield.get_shielded_actions(all_states[i])
                shielded_actions_list.append(allowable_actions)
                shielded_actions = tuple(set(shielded_actions)&set(allowable_actions))
        
        if action in shielded_actions:
            pass
        elif len(shielded_actions):
            action = np.random.choice(shielded_actions)

        if len(shielded_actions):
            if self.bouncedOffWall(state[0:2], action):
                for shielded_action in reversed(shielded_actions):
                    if not self.bouncedOffWall(state[0:2], shielded_action):
                        action = shielded_action
                        break
        else:
            print('No safe actions')
            pass

        return action

    def bouncedOffWall(self, rob_state, action):
        rob_state_next = self.trans(rob_state, action)
        if action != 0 and rob_state_next == rob_state:
            return True
        return False

    def trans_right(self, state):
        next_state_x = min(state[0] + 1, self.n - 1)
        next_state_y = state[1]
        
        return (next_state_x, next_state_y)
    
    def trans_left(self, state):
        next_state_x = max(state[0] - 1, 0)
        next_state_y = state[1]
        
        return (next_state_x, next_state_y)

    def trans_up(self, state):
        next_state_x = state[0]
        next_state_y = min(state[1] + 1, self.n - 1)
        
        return (next_state_x, next_state_y)

    def trans_down(self, state):
        next_state_x = state[0]
        next_state_y = max(state[1] - 1, 0)
        
        return (next_state_x, next_state_y)

    def trans_stay(self, state):
        return state

    def trans(self, state, action):
        if action < 0 or action >= self.n_action:
            raise ValueError("Action value out of bounds")
        if state[0] < 0 or state[0] >= self.n or state[1] < 0 or state[1] >= self.n:
            raise ValueError("State value out of bounds")
        
        if action == 1:
            return self.trans_up(state)
        if action == 2:
            return self.trans_right(state)
        if action == 3:
            return self.trans_down(state)
        if action == 4:
            return self.trans_left(state)
        if action == 0:
            return self.trans_stay(state) # Only for robot and not for adversary    

    def UpdateStateList(self, state):
        self.state_list = np.roll(self.state_list, -1, axis=0)
        self.state_list[-1] = np.array(state)

    def step(self, action):

        self.episode_steps += 1
        prev_state_rob = self.rob

        # The action is taken based on the current state and hence robot position should change first in step
        if self.shield_active:
            action_shield = 0
            prob_max = -1.0
            prob_list = []
            for i in range(5):
                obs_loc = self.state_list[0].tolist()
                prob = self.probDict[((True, obs_loc[0], obs_loc[1], self.action_pprev, self.action_prev, obs_loc[2], obs_loc[3]), i)]
                prob_list.append(prob)
                if prob > prob_max:
                    prob_max = prob
                    action_shield = i
            prob_action = self.probDict[((True, obs_loc[0], obs_loc[1], self.action_pprev, self.action_prev, obs_loc[2], obs_loc[3]), action)]
            if prob_action <= self.prob_threshold:
                if prob_action <= prob_max:
                    action = action_shield
            # print('===========================')
            # print(f'Obs: {obs_loc}; PPrev Action: {self.action_pprev}; Prev Action: {self.action_prev}')
            # print(f'Action to be taken: {action}; Action shield: {action_shield}')
            # print(f'Probabilities- All: {prob_list[0]:0.2f}, {prob_list[1]:0.2f}, {prob_list[2]:0.2f}, {prob_list[3]:0.2f}, {prob_list[4]:0.2f}, Action: {prob_action:0.2f}')
            # print(f'Action to be taken: {action}; Action shield: {action_shield}')
            
            # cur_rob_loc = tuple(self.state_list[-1,0:2].tolist())
            # pre_adv_loc = tuple(self.state_list[0,2:4].tolist())
            # cur_rob_pre_adv_loc = (*cur_rob_loc, *pre_adv_loc)
            # action = self.get_shielded_action(action, cur_rob_pre_adv_loc) 

        self.action_pprev = self.action_prev
        self.action_prev  = action

        self.rob = self.trans(self.rob, action)
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        if len(self.state_list == 1):
            obs = state
        else:
            obs = tuple(self.state_list[1].tolist() + [int(self.action_pprev), int(self.action_prev)]) #One step forward

        if self.rob == self.goal:
            info['gamestatus'] = 'won'
            reward = 2
            done   = True
            return obs, reward, done, info

        action_ghost = np.random.randint(1, self.n_action) # ghost cannot stay in place
        self.adv = self.trans(self.adv, action_ghost)
        
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        self.UpdateStateList(state)
        obs = tuple(self.state_list[0].tolist() + [int(self.action_pprev), int(self.action_prev)])
        
        if self.adv == self.rob:
            info['gamestatus'] = 'lost'
            reward = -10
            done   = True
            return obs, reward, done, info
        
        if math.dist(self.rob, self.goal) < math.dist(prev_state_rob, self.goal):
            reward = 0
        else: 
            reward = -1/self.max_episode_steps

        done = False
        if self.episode_steps >= self.max_episode_steps:
            done = True
            info['gamestatus'] = 'tie'

        info['gamestatus'] = 'none'
        
        return obs, reward, done, info

    def reset(self):
        self.rob = (0,0)
        self.adv  = (np.random.randint(int(self.n/3), int(3*self.n/3)), 
                     np.random.randint(int(self.n/3), int(3*self.n/3)))
        self.episode_steps = 0

        self.action_prev = int(0)
        self.action_pprev = int(0)

        state = (*self.rob, *self.adv)

        self.state_list = np.repeat(np.array(state, dtype=int).reshape(1,-1), self.time_delay + 1, axis=0)
        
        obs = tuple(self.state_list[0].tolist() + [self.action_pprev, self.action_prev])
        # return state
        return obs