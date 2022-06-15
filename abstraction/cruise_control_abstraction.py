import os 
import itertools

min_rel_vel = -30 
max_rel_vel = 30

max_rel_dist = 200
min_rel_dist = 0

min_fv_acc = -0.5 
max_fv_acc = 0.5 

min_ego_acc = -1.5 
max_ego_acc = 1.5 

num_rel_dist_states = 200 
num_rel_vel_states = 60

rel_dist_tuples = []
rel_vel_tuples = []
ego_acc_tuples = []

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

for i in range(int((max_rel_dist - min_rel_dist) / del_rel_dist)):
	rel_dist_tuples.append((min_rel_dist + i * del_rel_dist, min_rel_dist + (i + 1) * del_rel_dist))

for i in range(int((max_rel_vel - min_rel_vel) / del_rel_vel)):
	rel_vel_tuples.append((min_rel_vel + i * del_rel_vel, min_rel_vel + (i + 1) * del_rel_vel))

for i in range(int((max_ego_acc - min_ego_acc) / del_ego_acc)):
	ego_acc_tuples.append((min_ego_acc + i * del_ego_acc, min_ego_acc + (i + 1) * del_ego_acc))

cruise_control_mdp_transitions = []

for state_rel_dist in rel_dist_tuples:

	state_min_rel_dist = state_rel_dist[0] 
	state_max_rel_dist = state_rel_dist[1]

	for state_rel_vel in rel_vel_tuples:
	
		state_min_rel_vel = state_rel_vel[0]
		state_max_rel_vel = state_rel_vel[1]

		transition_list = []

		print("Generating transitions for the relative distance and relative velocity : ", state_rel_dist, state_rel_vel)
	
		for action in ego_acc_tuples:

			act_min = action[0]
			act_max = action[1]

			#print(state_rel_dist, state_rel_vel, action)

			"""
			### Relative distance propogation
			"""

			# calculating the minimum and maximum values for the next state relative distance

			next_state_min_rel_dist = state_min_rel_dist + state_min_rel_vel * del_t + 0.5 * (min_fv_acc - act_max) * del_t ** 2
			next_state_max_rel_dist = state_max_rel_dist + state_max_rel_vel * del_t + 0.5 * (max_fv_acc - act_min) * del_t ** 2 

			# calculating the index for the relative distance tuple corresponding to the minimum next state relative distance

			next_state_min_rel_dist_idx = -1 

			for i in range(len(rel_dist_tuples)):
				rel_dist_tup = rel_dist_tuples[i]
				if abs(next_state_min_rel_dist - rel_dist_tup[0]) < del_rel_dist:
					next_state_min_rel_dist_idx = i 
					break 

			# calculating the index for the relative distance tuple corresponding to the maximum next state relative distance

			next_state_max_rel_dist_idx = -1 

			for i in reversed(range(len(rel_dist_tuples))):
				rel_dist_tup = rel_dist_tuples[i]
				if abs(next_state_max_rel_dist - rel_dist_tup[1]) < del_rel_dist:
					next_state_max_rel_dist_idx = i 
					break 

			# the relative distance tuples that correspond to minimum and maximum next state relative distance

			next_states_rel_dist = rel_dist_tuples[next_state_min_rel_dist_idx : next_state_max_rel_dist_idx + 1]

			# adding the corner states if required 

			if next_state_min_rel_dist < min_rel_dist or next_state_max_rel_dist < min_rel_dist:
				next_states_rel_dist.append(starting_rel_dist_tuple) 

			if next_state_min_rel_dist > max_rel_dist or next_state_max_rel_dist > max_rel_dist:
				next_states_rel_dist.append(ending_rel_dist_tuple)

			"""
			### Relative velocity propogation
			"""

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

			# adding the corner states if required

			if next_state_min_rel_vel < min_rel_vel or next_state_max_rel_vel < min_rel_vel:
				next_states_rel_vel.append(starting_rel_vel_tuple)

			if next_state_min_rel_vel > max_rel_vel or next_state_max_rel_vel > max_rel_vel:
				next_states_rel_vel.append(ending_rel_vel_tuple)

			#print(next_state_min_rel_dist, next_state_max_rel_dist, next_state_min_rel_vel, next_state_max_rel_vel)
			#print(next_state_min_rel_dist_idx, next_state_max_rel_dist_idx, next_states_rel_dist, next_states_rel_vel)

			next_states = list(itertools.product(next_states_rel_dist, next_states_rel_vel))

			# storing the transition

			#print('############################################################')

			#print(state_rel_dist, state_rel_vel, action, next_state_min_rel_dist, next_state_max_rel_dist, next_state_min_rel_vel, next_state_max_rel_vel, next_states_rel_dist, next_states_rel_vel)

			transition = [state_rel_dist, state_rel_vel, action, next_states]
			#print("---------------------------------------------------------")
			#print(transition)   
			transition_list.append(transition)

		per_state_transition_mdp = [state_rel_dist, state_rel_vel, transition_list]

		cruise_control_mdp_transitions.append(per_state_transition_mdp)

"""
### converting interval states to labeled states
"""

transitions_with_state_labels = []



for mdp_transitions in cruise_control_mdp_transitions:

	rel_dist = mdp_transitions[0]
	rel_vel = mdp_transitions[1]

	lhs_rel_dist_label = rel_dist_tuples.index(rel_dist) 
	lhs_rel_vel_label = rel_vel_tuples.index(rel_vel) 

	rel_dist_str = "(s=%d)" % lhs_rel_dist_label
	rel_vel_str = "(v=%d)" % lhs_rel_vel_label

	print("generating transitions for rel_dist = " + rel_dist_str + " and rel_vel = " + rel_vel_str) 

	per_state_transitions = []

	for rel_dist, rel_vel, action, transitions in mdp_transitions[2]:
		#print("#########################")
		#print(rel_dist, rel_vel, action, transitions)

		action_label = ego_acc_tuples.index(action)

		potential_next_states = []
		for transition in transitions:
			potential_next_state = (rel_dist_tuples.index(transition[0]), rel_vel_tuples.index(transition[1]))
			potential_next_states.append(potential_next_state)
		
		potential_next_states = unique(potential_next_states)
		transition_tuple = (lhs_rel_dist_label, lhs_rel_vel_label, action_label, potential_next_states)
		per_state_transitions.append(transition_tuple)

	transitions_with_state_labels.append(per_state_transitions)

cruise_control_mdp_file = 'cruise_control.prism'
file = open(cruise_control_mdp_file, 'w+')

for state_transitions in transitions_with_state_labels:
	for state_action_transition in state_transitions:
		rel_dist_str = "(s=%d)" % state_action_transition[0]
		rel_vel_str = "(v=%d)" % state_action_transition[1]
		ego_acc_command = "[a" +  str(state_action_transition[2]) + "]"

		string = ""
		string += ego_acc_command
		string += rel_dist_str
		string += " & "
		string += rel_vel_str
		string += " -> "

		num_potential_transitions = len(state_action_transition[3])
		transition_prob = "1/%d" % num_potential_transitions

		i = 0
		for potential_next_state in state_action_transition[3]:
			next_rel_dist_str = "(s'=%d)" % potential_next_state[0]
			next_rel_vel_str = "(v'=%d)" % potential_next_state[1]

			string += transition_prob
			string += next_rel_dist_str
			string += " & "
			string += next_rel_vel_str

			if i < len(state_action_transition[3]) - 1:
				string += " + "

			i += 1
				
		string += ";"
		string += "\n"

		print(string)

		file.write(string)

	file.write('\n')

file.close()


