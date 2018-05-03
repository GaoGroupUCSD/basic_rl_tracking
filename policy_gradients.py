import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd
from env import *
from actor import *
from general_functions import *
import time
#from graphviz import Digraph
#import torchvision.models as models
#from torchviz import make_dot

LIMITS = (0, 9)
NUM_ACTIONS = 3
MAX_STEPS = 10

if __name__ == '__main__':
    environment = env()
    
    # TODO: Switch between the below 2 lines to begin training from scratch, or load from a previous checkpoint!
    #policy_network = actor()
    policy_network = torch.load('pg_checkpoints/800000.pkl')

    policy_optimizer = optim.SGD(policy_network.parameters(), lr=0.001)
    
    # TODO: Change this if you are continuing from a checkpoint!
    trajectory = 800000

    while True:

        actions = []
	policy_optimizer.zero_grad()

	s = environment.first_time_step()
	a_probs = policy_network(convert_to_variable(s))
	a = sample(a_probs)
	environment.perform_action(a)
	'''print representation(a)
	print 'Board: ' + str(environment.get_board())
	print 'Current box: ' + str(environment.get_box())
	print 'Probabilities: ' + str(a_probs)'''

	actions.append(a)
	action_probs = a_probs

	while not environment.done():
	    s = environment.next_time_step()
	    a_probs = policy_network(convert_to_variable(s))
	    a = sample(a_probs)
	    environment.perform_action(a)
	    '''print representation(a)
	    print 'Board: ' + str(environment.get_board())
	    print 'Current box: ' + str(environment.get_box())
	    print 'Probabilities: ' + str(a_probs)'''
	    
	    actions.append(a)
	    action_probs = torch.cat((action_probs, a_probs), 0)

	r = environment.get_reward() * 1.0

	#print 'Final reward: ' + str(r)

        # Compute expected returns from critic network
        discounted_rewards = compute_discounted_rewards(r, actions)

	# Compute gradients for policy and critic
	#loss = compute_policy_gradients(actions, action_probs, discounted_rewards)
	gradients = -compute_pg_gradients(actions, action_probs, discounted_rewards)

	#print 'Actor gradients: ' + str(policy_gradients)
	#print 'Critic gradients: ' + str(critic_gradients)

        # Training
	#loss.backward()
	action_probs.backward(gradients)

	#print_gradients(policy_network)

        # Visualize autograd computation graph
	#g = make_dot(loss)
	#g.view()
        
	policy_optimizer.step()

        if (trajectory + 1) % 100000 == 0:
            torch.save(policy_network, 'pg_checkpoints/' + str(trajectory + 1) + '.pkl')
	    print (trajectory + 1)

	trajectory += 1
	#print
	#print '-------------------------------------------------------------------------'
	#time.sleep(5)
