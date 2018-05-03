import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd
from env import *
from actor import *
from critic import *
from general_functions import *
import time

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

if __name__ == '__main__':
    environment = env()
    policy_network = actor()
    value_network = critic()
    critic_optimizer = optim.SGD(value_network.parameters(), lr=0.01)
    actor_optimizer = optim.SGD(policy_network.parameters(), lr=0.01)
    trajectory = 0

    while True:

        actions = []
	next_actions = []
        critic_optimizer.zero_grad()
	actor_optimizer.zero_grad()

	s = environment.first_time_step()
	q_values = value_network(convert_to_variable(s))
	a_probs = policy_network(convert_to_variable(s))
	a = sample(a_probs)
	#print representation(a)
	next_s = environment.perform_action(a)
	next_q_values = value_network(convert_to_variable(next_s))
	next_a_probs = policy_network(convert_to_variable(next_s))
	next_a = sample(next_a_probs)
	'''print 'Board: ' + str(environment.get_board())
	print 'Previous box: ' + str(environment.get_box(s))
	print 'Current box: ' + str(environment.get_box(next_s))
	print 'Q values: ' + str(q_values)
	print 'Probabilities: ' + str(a_probs)'''

	q_states = q_values
	q_next_states = next_q_values
	actions.append(a)
	next_actions.append(next_a)
	action_probs = a_probs

	while not environment.done():
	    s = environment.next_time_step()
	    q_values = value_network(convert_to_variable(s))
	    a_probs = policy_network(convert_to_variable(s))
	    a = sample(a_probs)
	    #print representation(a)

	    next_s = environment.perform_action(a)
	    next_q_values = value_network(convert_to_variable(next_s))
	    next_a_probs = policy_network(convert_to_variable(next_s))
	    next_a = sample(next_a_probs)

	    '''print 'Board: ' + str(environment.get_board())
	    print 'Previous box: ' + str(environment.get_box(s))
	    print 'Current box: ' + str(environment.get_box(next_s))
	    print 'Q values: ' + str(q_values)
	    print 'Probabilities: ' + str(a_probs)'''
	    
	    q_states = torch.cat((q_states, q_values), 0)
	    q_next_states = torch.cat((q_next_states, next_q_values), 0)
	    actions.append(a)
	    next_actions.append(next_a)
	    action_probs = torch.cat((action_probs, a_probs), 0)

	r = np.zeros(len(q_values))
	r[-1] = environment.get_reward() * 1.0

	#print 'Final reward: ' + str(r[-1])

        # Compute expected returns from critic network
        target_values = calculate_target_values_AC(r, q_states, q_next_states, actions, next_actions)

	# Compute gradients for actor and critic
	actor_gradients = compute_actor_gradients(actions, action_probs, q_states)
	critic_gradients = compute_critic_gradients(q_states, target_values)

	#print 'Actor gradients: ' + str(actor_gradients)
	#print 'Critic gradients: ' + str(critic_gradients)

        # Training

	# TODO: update rules left!
        policy_network._grad = actor_gradients
	value_network._grad = critic_gradients
	critic_optimizer.step()
	actor_optimizer.step()

        if (trajectory + 1) % 100000 == 0:
            torch.save(policy_network, 'ac_checkpoints/policy_' + str(trajectory + 1) + '.pkl')
	    torch.save(value_network, 'ac_checkpoints/value_' + str(trajectory + 1) + '.pkl')
	    print (trajectory + 1)

	trajectory += 1
	#print
	#time.sleep(5)
