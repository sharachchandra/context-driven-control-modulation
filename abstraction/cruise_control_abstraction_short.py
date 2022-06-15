import os 
import itertools

import numpy as np

def unique(list1):
	unique_list = []
	for x in list1:
		if x not in unique_list:
			unique_list.append(x)

	return unique_list

min_rel_vel = -30
max_rel_vel = 30

max_rel_dist = 100
min_rel_dist = 0

min_fv_acc = -0.5 
max_fv_acc = 0.5 

min_ego_acc = -1.5 
max_ego_acc = 1.5 

del_rel_dist = 1.0 
del_rel_vel = 1.0 
del_ego_acc = 0.1
del_t = 1.0

rel_dist_tuples = []
rel_vel_tuples = []
ego_acc_tuples = []

for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

for i in range(int((max_rel_vel - min_rel_vel) / del_rel_vel)):
	rel_vel_tuples.append((min_rel_vel + i * del_rel_vel, min_rel_vel + (i + 1) * del_rel_vel))

for i in range(int((max_ego_acc - min_ego_acc) / del_ego_acc)):
	ego_acc_tuples.append((min_ego_acc + i * del_ego_acc, min_ego_acc + (i + 1) * del_ego_acc))

rel_dist_pts = np.arange(min_rel_dist, max_rel_dist, del_rel_dist) 
rel_vel_pts = np.arange(min_rel_vel, max_rel_vel, del_rel_vel) 

cruise_control_mdp_file = 'cruise_control.prism'
file = open(cruise_control_mdp_file, 'w+')

for state_rel_dist in rel_dist_tuples:

	state_min_rel_dist = state_rel_dist[0] 
	state_max_rel_dist = state_rel_dist[1]

	rel_dist_idx = rel_dist_tuples.index(state_rel_dist)
	rel_dist_str = "(s=%d)" % rel_dist_idx

	for state_rel_vel in rel_vel_tuples:
	
		state_min_rel_vel = state_rel_vel[0]
		state_max_rel_vel = state_rel_vel[1]

		rel_vel_idx = rel_vel_tuples.index(state_rel_vel)
		rel_vel_str = "(v=%d)" % rel_vel_idx 

		transition_list = []
	
		for action in ego_acc_tuples:

			act_min = action[0]
			act_max = action[1]

			act_idx = ego_acc_tuples.index(action)
			ego_acc_str = "[a" +  str(act_idx) + "]" 

			string = ""
			string += ego_acc_str
			string += rel_dist_str
			string += " & "
			string += rel_vel_str
			string += " -> "

			# calculating the minimum and maximum values for the next state relative distance

			next_state_min_rel_dist = state_min_rel_dist + state_min_rel_vel * del_t + 0.5 * (min_fv_acc - act_max) * del_t ** 2
			next_state_max_rel_dist = state_max_rel_dist + state_max_rel_vel * del_t + 0.5 * (max_fv_acc - act_min) * del_t ** 2 

			# calculating the minimum and maximum values for the next state relative velocity

			next_state_min_rel_vel = state_min_rel_vel + (min_fv_acc - act_max)
			next_state_max_rel_vel = state_max_rel_vel + (max_fv_acc - act_min)

			# calculating the index for the relative distance tuple corresponding to the minimum next state relative distance 

			list1 = [abs(next_state_min_rel_dist - rel_dist_pts[i]) for i in range(len(rel_dist_pts))]
			loc1 = list1.index(min(list1))

			if loc1 > next_state_min_rel_dist and next_state_min_rel_dist > min_rel_dist:
				loc1 -= del_rel_dist 

			list2 = [abs(next_state_max_rel_dist - rel_dist_pts[i]) for i in range(len(rel_dist_pts))]
			loc2 = list2.index(min(list2))

			if loc2 < next_state_max_rel_dist and next_state_max_rel_dist > min_rel_dist:
				loc2 += del_rel_dist 

			indices = list(np.arange(int(loc1), int(loc2)))

			if len(indices) == 0:
				if loc1 == 0 or loc2 == 0:
					indices.append(0)

			indices = unique(indices)
			next_states_rel_dist = [rel_dist_tuples[idx] for idx in indices] 
			#print("###")
			#print(next_state_min_rel_dist, next_state_max_rel_dist, action, loc1, loc2, next_states_rel_dist, indices)
			next_rel_dist_idxs = indices

			# calculating the index for the relative velocity tuple corresponding to the minimum next state relative velocity

			list1 = [abs(next_state_min_rel_vel - rel_vel_pts[i]) for i in range(len(rel_vel_pts))]
			loc1 = list1.index(min(list1))

			if rel_vel_pts[loc1] > next_state_min_rel_vel and next_state_min_rel_vel > min_rel_vel:
				loc1 -= del_rel_vel 

			list2 = [abs(next_state_max_rel_vel - rel_vel_pts[i]) for i in range(len(rel_vel_pts))]
			loc2 = list2.index(min(list2))

			if rel_vel_pts[loc2] < next_state_max_rel_vel and next_state_max_rel_vel > min_rel_vel:
				loc2 += del_rel_vel 

			indices = list(np.arange(int(loc1), int(loc2)))

			if len(indices) == 0 and (loc1 == 0 or loc2 == 0):
				indices.append(0)

			indices = unique(indices)
			next_states_rel_vel = [rel_vel_tuples[idx] for idx in indices] 
			next_rel_vel_idxs = indices
			"""
			cruise control example 
			Experiments 
			1. 
			2. 
			"""
			#print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
			#print(next_state_min_rel_vel, next_state_max_rel_vel, action, loc1, loc2, next_states_rel_vel, indices)

			num_potential_transitions = len(next_states_rel_dist) * len(next_states_rel_vel)
			transition_prob = "1/%d" % num_potential_transitions

			i = 0
			for next_rel_dist_idx in next_rel_dist_idxs:
				for next_rel_vel_idx in next_rel_vel_idxs:

					next_rel_dist_str = "(s'=%d)" % next_rel_dist_idx
					next_rel_vel_str = "(v'=%d)" % next_rel_vel_idx

					string += transition_prob
					string += ":"
					string += next_rel_dist_str
					string += " & "
					string += next_rel_vel_str

					if i < num_potential_transitions - 1:
						string += " + "

					i += 1
				
			string += ";"
			string += "\n"

			print(string)

			file.write(string)

		file.write('\n')

file.close()
		