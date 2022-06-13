import os 
import itertools

import numpy as np

min_rel_vel = -3 
max_rel_vel = 3

max_rel_dist = 20
min_rel_dist = 0

min_fv_acc = -0.5 
max_fv_acc = 0.5 

min_ego_acc = -1.5 
max_ego_acc = 1.5 

num_rel_dist_states = 200 
num_rel_vel_states = 60

del_rel_dist = 1.0 
del_rel_vel = 1.0 
del_ego_acc = 0.1
del_t = 1.0

starting_rel_dist_tuple = (min_rel_dist, min_rel_dist + del_rel_dist) 
ending_rel_dist_tuple = (max_rel_dist - del_rel_dist, max_rel_dist)

starting_rel_vel_tuple = (min_rel_vel, min_rel_vel + del_rel_vel) 
ending_rel_vel_tuple = (max_rel_vel - del_rel_vel, max_rel_vel)

def unique(list1):
	unique_list = []
	for x in list1:
		if x not in unique_list:
			unique_list.append(x)

	return unique_list

rel_dist_tuples = []
rel_vel_tuples = []
ego_acc_tuples = []

for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

for i in range(int((max_rel_vel - min_rel_vel) / del_rel_vel)):
	rel_vel_tuples.append((min_rel_vel + i * del_rel_vel, min_rel_vel + (i + 1) * del_rel_vel))

for i in range(int((max_ego_acc - min_ego_acc) / del_ego_acc)):
	ego_acc_tuples.append((min_ego_acc + i * del_ego_acc, min_ego_acc + (i + 1) * del_ego_acc))

rel_dist_pts = np.arange(min_rel_dist, max_rel_dist + del_rel_dist, del_rel_dist) 
rel_vel_pts = np.arange(min_rel_vel, max_rel_vel + del_rel_vel, del_rel_vel) 

#print(rel_dist_pts, rel_dist_tuples)
#print(rel_vel_pts, rel_vel_tuples)

cruise_control_mdp_transitions = []

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

		#print("Generating transitions for the relative distance and relative velocity : ", state_rel_dist, state_rel_vel)
	
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

			# calculating the index for the relative distance tuple corresponding to the minimum next state relative distance 

			list1 = [abs(next_state_min_rel_dist - rel_dist_pts[i]) for i in range(len(rel_dist_pts))]
			loc1 = rel_dist_pts[list1.index(min(list1))]

			if loc1 > next_state_min_rel_dist and next_state_min_rel_dist > 0:
				loc1 -= del_rel_dist 

			next_state_min_rel_dist_idx_dummy = int(loc1)

			#next_state_min_rel_dist_idx_dummy = np.argmin(np.square([next_state_min_rel_dist - rel_dist_pts[i] for i in range(len(rel_dist_pts))])

			next_state_min_rel_dist_idx = -1 

			for i in range(len(rel_dist_tuples)):
				rel_dist_tup = rel_dist_tuples[i]
				if abs(next_state_min_rel_dist - rel_dist_tup[0]) < del_rel_dist:
					next_state_min_rel_dist_idx = i 
					break 

			print(next_state_min_rel_dist, next_state_min_rel_dist_idx_dummy, next_state_min_rel_dist_idx)

			# calculating the index for the relative distance tuple corresponding to the maximum next state relative distance

			next_state_max_rel_dist_idx = -1 

			for i in reversed(range(len(rel_dist_tuples))):
				rel_dist_tup = rel_dist_tuples[i]
				if abs(next_state_max_rel_dist - rel_dist_tup[1]) < del_rel_dist:
					next_state_max_rel_dist_idx = i 
					break 

			# the relative distance tuples that correspond to minimum and maximum next state relative distance

			next_states_rel_dist = rel_dist_tuples[next_state_min_rel_dist_idx : next_state_max_rel_dist_idx + 1]
			next_rel_dist_idxs = [rel_dist_tuples.index(next_states_rel_dist[i]) for i in range(len(next_states_rel_dist))]

			# adding the corner states if required 

			if next_state_min_rel_dist < min_rel_dist or next_state_max_rel_dist < min_rel_dist:
				next_states_rel_dist.append(starting_rel_dist_tuple)
				next_rel_dist_idxs.append(rel_dist_tuples.index(starting_rel_dist_tuple)) 

			if next_state_min_rel_dist > max_rel_dist or next_state_max_rel_dist > max_rel_dist:
				next_states_rel_dist.append(ending_rel_dist_tuple)
				next_rel_dist_idxs.append(rel_dist_tuples.index(ending_rel_dist_tuple)) 

			next_rel_dist_idxs = unique(next_rel_dist_idxs)

			# calculating the minimum and maximum values for the next state relative velocity

			next_state_min_rel_vel = state_min_rel_vel + (min_fv_acc - act_max)
			next_state_max_rel_vel = state_max_rel_vel + (max_fv_acc - act_min)

			# calculating the index for the relative distance tuple corresponding to the minimum next state relative velocity

			next_state_min_rel_vel_idx = -1 

			for i in range(len(rel_vel_tuples)):
				rel_vel_tup = rel_vel_tuples[i]
				if abs(next_state_min_rel_vel - rel_vel_tup[0]) < del_rel_vel:
					next_state_min_rel_vel_idx = i 
					break  

			# calculating the index for the relative distance tuple corresponding to the maximum next state relative velocity

			next_state_max_rel_vel_idx = -1 

			for i in reversed(range(len(rel_vel_tuples))):
				rel_vel_tup = rel_vel_tuples[i]
				if abs(next_state_max_rel_vel - rel_vel_tup[1]) < del_rel_vel:
					next_state_max_rel_vel_idx = i 
					break            

			# the relative distance tuples that correspond to minimum and maximum next state relative velocities

			next_states_rel_vel = rel_vel_tuples[next_state_min_rel_vel_idx : next_state_max_rel_vel_idx + 1]
			next_rel_vel_idxs = [rel_vel_tuples.index(next_states_rel_vel[i]) for i in range(len(next_states_rel_vel))]
			

			# adding the corner states if required

			if next_state_min_rel_vel < min_rel_vel or next_state_max_rel_vel < min_rel_vel:
				next_states_rel_vel.append(starting_rel_vel_tuple)
				next_rel_vel_idxs.append(rel_vel_tuples.index(starting_rel_vel_tuple)) 

			if next_state_min_rel_vel > max_rel_vel or next_state_max_rel_vel > max_rel_vel:
				next_states_rel_vel.append(ending_rel_vel_tuple)
				next_rel_vel_idxs.append(rel_vel_tuples.index(ending_rel_vel_tuple)) 

			#print(next_states_rel_vel, next_rel_vel_idxs)

			next_rel_vel_idxs = unique(next_rel_vel_idxs) 

			num_potential_transitions = len(next_rel_dist_idxs) * len(next_rel_vel_idxs)
			transition_prob = "1/%d" % num_potential_transitions

			i = 0
			for next_rel_dist_idx in next_rel_dist_idxs:
				for next_rel_vel_idx in next_rel_vel_idxs:

					next_rel_dist_str = "(s'=%d)" % next_rel_dist_idx
					next_rel_vel_str = "(v'=%d)" % next_rel_vel_idx

					string += transition_prob
					string += next_rel_dist_str
					string += " & "
					string += next_rel_vel_str

					if i < num_potential_transitions - 1:
						string += " + "

					i += 1
				
			string += ";"
			string += "\n"

			#print(string)
		break
