from gym.envs.registration import register

register(id='gridworld-v0', entry_point = 'gym_gridworld.envs:Gridworld')
register(id='gridworld-v1', entry_point = 'gym_gridworld.envs:Gridworld1')
register(id='gridworld-v2', entry_point = 'gym_gridworld.envs:Gridworld2')
register(id='gridworld-v3', entry_point = 'gym_gridworld.envs:Gridworld3')
register(id='gridworld-v4', entry_point = 'gym_gridworld.envs:Gridworld4')