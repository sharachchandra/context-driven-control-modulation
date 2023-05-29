import gym
import numpy as np

class Gridworld(gym.Env):
    def __init__(self) -> None:
        super().__init__()
    
        self.n_action = 5
        self.action_space = gym.spaces.Discrete(self.n_action)
        
        self.n = 10
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n),
                                                   gym.spaces.Discrete(self.n)))
        # (row, column)
        self.max_episode_steps = 40
        self.goal = (9,9)
    
        self.shield = self.Shield()
        self.shield_active = False

    class Shield():
        def __init__(self) -> None:
            pass
        
        def shielded_actions(self, state):
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
            
            else:
                return (0, 1, 2, 3, 4)
    
    def activate_shield(self, val = True):
        self.shield_active = val

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
            print(state)
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
            return self.trans_stay(state) # Only for robot and not for adversary (adversary)

    def step(self, action):

        self.episode_steps += 1

        # The action is taken based on the current state and hence robot position should change first in step
        state  = (*self.rob, *self.adv)
        shielded_actions = self.shield.shielded_actions(state)
        if self.shield_active:
            if action in shielded_actions:
                pass
            elif len(shielded_actions):
                action = np.random.choice(shielded_actions)

        self.rob = self.trans(self.rob, action)
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        state  = (*self.rob, *self.adv)
        if self.rob == self.goal:
            print('game won')
            reward = 1
            done   = True
            return state, reward, done, info

        action_ghost = np.random.randint(1, self.n_action) # ghost cannot stay in place
        self.adv = self.trans(self.adv, action_ghost)
        
        state  = (*self.rob, *self.adv)
        info   = {'robot': self.rob,
                  'adversary' : self.adv}
        if self.adv == self.rob:
            # print('collided')
            reward = -1
            done   = True
            return state, reward, done, info
        
        reward = -1/self.max_episode_steps
        done = False
        if self.episode_steps >= self.max_episode_steps:
            done = True

        return state, reward, done, info

    def reset(self):
        self.rob = (0,0)
        self.adv  = (np.random.randint(0, self.n), np.random.randint(0, self.n))
        self.episode_steps = 0
        state = (*self.rob, *self.adv)
        return state