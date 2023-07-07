import numpy as np
prob_dict = np.load('shields_RAL/Qmax_values_0_td.npy', allow_pickle = True).item()
# prob_dict = np.load('shields/state_action_values_1_td.npy', allow_pickle = True).item()
print(prob_dict.keys())
# print(prob_dict[((0, 0, 6, 7, 0, 1), 0)])
print(type(prob_dict))
num_xbins=8
def convert_state_to_int(state):
	increments = [(num_xbins**3)*2, (num_xbins**2)*2, num_xbins*2, 2, 1]
	return np.sum(np.multiply(list(state), increments))
state = (0,0,0,2,1)
print(convert_state_to_int(state))